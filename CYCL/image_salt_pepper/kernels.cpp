#include <CL/sycl.hpp>
#include <cmath>

using namespace cl::sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
    float thredshold, int window_size,
    int height, int width) {

    int ws2 = (window_size - 1) / 2; // Calcula el radio de la ventana
    int window_area = window_size * window_size;

    Q.submit([&](handler& h) {
        h.parallel_for(range<2>(height, width), [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];

            // Si es un elemento borde copiamos su valor
            if (i < ws2 || i >= height - ws2 || j < ws2 || j >= width - ws2) {
                image_out[i * width + j] = im[i * width + j];
                return;
            }

            float window[9]; 
            int count = 0;
            for (int ii = -ws2; ii <= ws2; ++ii) {
                for (int jj = -ws2; jj <= ws2; ++jj) {
                    int row = i + ii;
                    int col = j + jj;
                    window[count++] = im[row * width + col];
                }
            }

            //Ordenamos los valores vecinos
            for (int k = 0; k < window_area - 1; ++k) {
                for (int l = 0; l < window_area - k - 1; ++l) {
                    if (window[l] > window[l + 1]) {
                        float temp = window[l];
                        window[l] = window[l + 1];
                        window[l + 1] = temp;
                    }
                }
            }

            float median = window[window_area / 2];
            float current = im[i * width + j];
            float diff = abs(median - current);
            //Seleccionamos el valor final dependiendo de la diferencia y el umbral
            if (diff <= thredshold) {
                image_out[i * width + j] = current;
            } else {
                image_out[i * width + j] = median;
            }
        });
    }).wait(); 
}