#include <CL/sycl.hpp>
#include <cmath>

using namespace cl::sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
    float thredshold, int window_size,
    int height, int width) {

    int rad = (window_size - 1) / 2; // Calcula el radio 
    int window_area = window_size * window_size;

    Q.submit([&](handler& h) {
        h.parallel_for(range<2>(height, width), [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];


            if (i < rad || i >= height - rad || j < rad || j >= width - rad) {
                image_out[i * width + j] = im[i * width + j];
                return;
            }

            float window[9]; 
            int count = 0;
            for (int ii = -rad; ii <= rad; ++ii) {
                for (int jj = -rad; jj <= rad; ++jj) {
                    int row = i + ii;
                    int col = j + jj;
                    window[count++] = im[row * width + col];
                }
            }

            for (int k = 0; k < window_area - 1; ++k) {
                for (int l = 0; l < window_area - k - 1; ++l) {
                    if (window[l] > window[l + 1]) {
                        float temp = window[l];
                        window[l] = window[l + 1];
                        window[l + 1] = temp;
                    }
                }
            }

            float mediana = window[window_area / 2];
            float val = im[i * width + j];
            if ((abs(mediana - val)) <= thredshold) {
                image_out[i * width + j] = val;
            } else {
                image_out[i * width + j] = mediana;
            }
        });
    }).wait(); 
}