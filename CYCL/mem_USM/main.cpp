#include <CL/sycl.hpp>

using  namespace  cl::sycl;

int main(int argc, char **argv) {

	if (argc!=2)  {
		std::cout << "./exec N"<< std::endl;
		return(-1);
	}

	int N = atoi(argv[1]);

	sycl::queue Q(sycl::gpu_selector_v);

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;


	// a in USM
	//float *a; // To fill with malloc_shared
	//Buff&ACC: std::vector<float> a(N);
	float *a = malloc_shared<float>(N,Q);//USM

	// Parallel for
	for(int i=0; i<N; i++)
		a[i] = i; // Init a

	//Buff&ACC: buffer buffer_a{a};

	// Create a command_group to issue command to the group
	Q.submit([&](handler &h) {
		//Buff&ACC: accessor acc_a{buffer_a, h ,read_write};

		// Submit the kernel
		h.parallel_for(N, [=](id<1> i) {
			a[i]*=3.0f;
		}); // End of the kernel function

	}).wait();       // End of the queue commands we waint on the event reported.

	//Buff&ACC: host_accessor a_(buffer_a,read_only);
	
	for(int i=0; i<N; i++)
		std::cout << "a[" << i << "] = " << a[i] << std::endl;


	free(a,Q);
  return 0;
}
