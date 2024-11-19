#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// Intervalos de 15mins en segundos
#define INTERVAL_SECONDS 900

// Categorias de aplicaciones de luz
#define OFFICE_MIN 300.0f
#define READING_MIN 500.0f
#define FACTORY_MIN 1000.0f
#define COUNTERPRODUCTIVE_MIN 5000.0f

// CUDA Kernel: Estadisticas por intervalo
__global__ void computeIntervalStats(
    const float *data, const int *intervals, int dataSize,
    float *tempSums, int *tempCounts,
    float *luxSums, float *luxSqSums, int *luxCounts)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < dataSize)
    {
        int intervalIdx = intervals[idx];

        printf("Hilo %d: Procesando en intervalo %d\n", idx, intervalIdx);

        // Actualización de estadisticas de intervalo
        atomicAdd(&tempSums[intervalIdx], data[idx * 3 + 1]);                      // Suma temperaturas
        atomicAdd(&tempCounts[intervalIdx], 1);                                    // Conteo temperaturas
        atomicAdd(&luxSums[intervalIdx], data[idx * 3 + 2]);                       // Suma lux
        atomicAdd(&luxSqSums[intervalIdx], data[idx * 3 + 2] * data[idx * 3 + 2]); // Suma cuadrados de lux (stdev)
        atomicAdd(&luxCounts[intervalIdx], 1);                                     // Conteo lux
    }
}

// CUDA Kernel: Calculo de Estadisticas Totales por Intervalo
__global__ void finalizeIntervalStats(
    const float *tempSums, const int *tempCounts, float *tempAverages,
    const float *luxSums, const float *luxSqSums, const int *luxCounts,
    float *luxStdDevs, int numIntervals)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numIntervals)
    {

        // Promedio de Temperatura
        if (tempCounts[idx] > 0)
        {
            tempAverages[idx] = tempSums[idx] / tempCounts[idx];
        }

        // Desviación estándar de lux
        if (luxCounts[idx] > 0)
        {
            float mean = luxSums[idx] / luxCounts[idx];
            float meanSq = luxSqSums[idx] / luxCounts[idx];
            luxStdDevs[idx] = sqrtf(meanSq - (mean * mean));
        }

        // Mostrar que hilo maneja qué intervalo
        printf("Hilo %d: Finalizando intervalo %d\n", idx, idx);
    }
}

// CUDA Kernel: Tiempo pasado en cada categoria de iluminacion
__global__ void calculateLuxCategoryTime(
    const float *lux, const float *timeDeltas,
    float *timeOffice, float *timeReading, float *timeFactory, float *timeCounterproductive, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        float deltaTime = timeDeltas[idx];

        // Agregar delta tiempo a cada categoría de iluminación
        if (lux[idx] >= OFFICE_MIN)
            atomicAdd(timeOffice, deltaTime);
        if (lux[idx] >= READING_MIN)
            atomicAdd(timeReading, deltaTime);
        if (lux[idx] >= FACTORY_MIN)
            atomicAdd(timeFactory, deltaTime);
        if (lux[idx] >= COUNTERPRODUCTIVE_MIN)
            atomicAdd(timeCounterproductive, deltaTime);

        // Mostrar que hilo maneja qué valor
        printf("Hilo %d: Procesando lux %.2f con delta tiempo %.2f\n", idx, lux[idx], deltaTime);
    }
}

// Parseo de timestamps a segundos
int parseTimestampToSeconds(const string &timestamp)
{
    int hours = 0, minutes = 0, seconds = 0;
    char delimiter;
    stringstream ss(timestamp);

    ss >> hours >> delimiter >> minutes >> delimiter >> seconds;
    if (ss.fail())
    {
        throw invalid_argument("Formato de timestamp inválido: " + timestamp);
    }

    return hours * 3600 + minutes * 60 + seconds;
}

// Lectura / parsing de CSV
vector<vector<float>> readCSV(const string &filename)
{
    ifstream file(filename);
    vector<vector<float>> data;
    string line;

    if (!file.is_open())
    {
        throw runtime_error("Error abriendo archivo: " + filename);
    }

    getline(file, line); // Saltar encabezado

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream lineStream(line);
        string cell;
        vector<float> row;

        try
        {
            getline(lineStream, cell, ',');
            row.push_back(static_cast<float>(parseTimestampToSeconds(cell)));

            getline(lineStream, cell, ',');
            row.push_back(stof(cell));

            getline(lineStream, cell, ',');
            row.push_back(stof(cell));

            data.push_back(row);
        }
        catch (const invalid_argument &e)
        {
            cerr << "Error parseando línea: " << e.what() << endl;
        }
    }

    return data;
}

int main()
{
    const string filename = "data.csv";

    try
    {
        // Leer y parsear el archivo CSV
        auto data = readCSV(filename);

        if (data.empty())
        {
            cerr << "Error: No se encontraron datos en el archivo CSV." << endl;
            return -1;
        }

        // Preparar datos y calcular índices de intervalos
        int dataSize = data.size();
        vector<float> flatData(dataSize * 3);
        vector<int> intervalIndices(dataSize);

        int startTime = static_cast<int>(data[0][0]);
        for (int i = 0; i < dataSize; ++i)
        {
            flatData[i * 3] = data[i][0];
            flatData[i * 3 + 1] = data[i][1];
            flatData[i * 3 + 2] = data[i][2];
            intervalIndices[i] = static_cast<int>((data[i][0] - startTime) / INTERVAL_SECONDS);
        }

        int numIntervals = intervalIndices.back() + 1;

        // Asignar memoria en GPU
        float *d_data, *d_tempSums, *d_tempAverages, *d_luxSums, *d_luxSqSums, *d_luxStdDevs;
        float *d_timeOffice, *d_timeReading, *d_timeFactory, *d_timeCounterproductive;
        int *d_intervals, *d_tempCounts, *d_luxCounts;

        cudaMalloc(&d_data, flatData.size() * sizeof(float));
        cudaMalloc(&d_intervals, intervalIndices.size() * sizeof(int));
        cudaMalloc(&d_tempSums, numIntervals * sizeof(float));
        cudaMalloc(&d_tempAverages, numIntervals * sizeof(float));
        cudaMalloc(&d_tempCounts, numIntervals * sizeof(int));
        cudaMalloc(&d_luxSums, numIntervals * sizeof(float));
        cudaMalloc(&d_luxSqSums, numIntervals * sizeof(float));
        cudaMalloc(&d_luxStdDevs, numIntervals * sizeof(float));
        cudaMalloc(&d_luxCounts, numIntervals * sizeof(int));
        cudaMalloc(&d_timeOffice, sizeof(float));
        cudaMalloc(&d_timeReading, sizeof(float));
        cudaMalloc(&d_timeFactory, sizeof(float));
        cudaMalloc(&d_timeCounterproductive, sizeof(float));

        // Inicializar memoria en GPU
        cudaMemcpy(d_data, flatData.data(), flatData.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_intervals, intervalIndices.data(), intervalIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_tempSums, 0, numIntervals * sizeof(float));
        cudaMemset(d_luxSums, 0, numIntervals * sizeof(float));
        cudaMemset(d_luxSqSums, 0, numIntervals * sizeof(float));
        cudaMemset(d_tempCounts, 0, numIntervals * sizeof(int));
        cudaMemset(d_luxCounts, 0, numIntervals * sizeof(int));
        cudaMemset(d_timeOffice, 0, sizeof(float));
        cudaMemset(d_timeReading, 0, sizeof(float));
        cudaMemset(d_timeFactory, 0, sizeof(float));
        cudaMemset(d_timeCounterproductive, 0, sizeof(float));

        // Calcular estadísticas por intervalo
        int threads = 256;
        int blocks = (dataSize + threads - 1) / threads;
        computeIntervalStats<<<blocks, threads>>>(d_data, d_intervals, dataSize, d_tempSums, d_tempCounts, d_luxSums, d_luxSqSums, d_luxCounts);

        // Finalizar estadísticas por intervalo
        blocks = (numIntervals + threads - 1) / threads;
        finalizeIntervalStats<<<blocks, threads>>>(d_tempSums, d_tempCounts, d_tempAverages, d_luxSums, d_luxSqSums, d_luxCounts, d_luxStdDevs, numIntervals);

        // Preparar datos para calcular tiempo en categorías de iluminación
        vector<float> lux(dataSize), timeDeltas(dataSize);
        for (size_t i = 1; i < data.size(); ++i)
        {
            lux[i] = data[i][2];
            timeDeltas[i] = data[i][0] - data[i - 1][0];
        }

        float *d_lux, *d_timeDeltas;
        cudaMalloc(&d_lux, lux.size() * sizeof(float));
        cudaMalloc(&d_timeDeltas, timeDeltas.size() * sizeof(float));
        cudaMemcpy(d_lux, lux.data(), lux.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_timeDeltas, timeDeltas.data(), timeDeltas.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Calcular tiempos en categorías de iluminación
        blocks = (lux.size() + threads - 1) / threads;
        calculateLuxCategoryTime<<<blocks, threads>>>(d_lux, d_timeDeltas, d_timeOffice, d_timeReading, d_timeFactory, d_timeCounterproductive, lux.size());

        cudaDeviceSynchronize();

        // Copiar resultados al host
        vector<float> tempAverages(numIntervals), luxStdDevs(numIntervals);
        float timeOffice, timeReading, timeFactory, timeCounterproductive;

        cudaMemcpy(tempAverages.data(), d_tempAverages, numIntervals * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(luxStdDevs.data(), d_luxStdDevs, numIntervals * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&timeOffice, d_timeOffice, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&timeReading, d_timeReading, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&timeFactory, d_timeFactory, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&timeCounterproductive, d_timeCounterproductive, sizeof(float), cudaMemcpyDeviceToHost);

        // Imprimir resultados
        for (int i = 0; i < numIntervals; ++i)
        {
            cout << "Intervalo " << i << " (Tiempo: " << i * INTERVAL_SECONDS
                      << " - " << (i + 1) * INTERVAL_SECONDS << " segundos):\n";
            cout << "  Temperatura promedio: " << tempAverages[i] << "°C\n";
            cout << "  Desviación estándar de lux: " << luxStdDevs[i] << "\n";
        }

        cout << "\nTiempo en categorías de iluminación:\n";
        cout << "  Iluminación de oficina (300 lux+): " << timeOffice << " segundos\n";
        cout << "  Iluminación de lectura (500 lux+): " << timeReading << " segundos\n";
        cout << "  Iluminación de fábrica (1000 lux+): " << timeFactory << " segundos\n";
        cout << "  Iluminación contraproducente (5000 lux+): " << timeCounterproductive << " segundos\n";

        // Liberar memoria en GPU
        cudaFree(d_data);
        cudaFree(d_intervals);
        cudaFree(d_tempSums);
        cudaFree(d_tempCounts);
        cudaFree(d_tempAverages);
        cudaFree(d_luxSums);
        cudaFree(d_luxSqSums);
        cudaFree(d_luxCounts);
        cudaFree(d_luxStdDevs);
        cudaFree(d_timeOffice);
        cudaFree(d_timeReading);
        cudaFree(d_timeFactory);
        cudaFree(d_timeCounterproductive);
        cudaFree(d_lux);
        cudaFree(d_timeDeltas);
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
