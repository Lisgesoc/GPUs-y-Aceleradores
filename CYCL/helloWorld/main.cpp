#include <CL/sycl.hpp>

using  namespace  cl::sycl;

class CustomDeviceSelector : public sycl::device_selector
{
public:
	int operator()(const sycl::device &dev) const override
	{
		std::string vendor = dev.get_info<sycl::info::device::vendor>();
		if (dev.is_gpu())
		{
			//std::cout << "GPU: " << dev.get_info<sycl::info::device::name>() << " (Vendor: " << vendor << ")\n";
			return (vendor.find("NVIDIA") != std::string::npos) ? 3 : 2;
		}
		else if (dev.is_cpu())
		{
			//std::cout << "CPU: " << dev.get_info<sycl::info::device::name>() << "\n";
			return 1;
		}
		else
		{
			//std::cout << "Other: " << dev.get_info<sycl::info::device::name>() << "\n";
			return 0;
		}
	}
};

int main(int argc, char **argv) {
	CustomDeviceSelector my_device_selector;

	sycl::queue Q(my_device_selector);

	std::cout << "Running on "
		<< Q.get_device().get_info<sycl::info::device::name>()
		<< std::endl;

	Q.submit([&](handler &cgh) {
		// Create a output stream
		sycl::stream sout(1024, 256, cgh);
		// Submit a unique task, using a lambda
		cgh.single_task([=]() {
			sout << "Hello, World!" << sycl::endl;
		}); // End of the kernel function
	});   // End of the queue commands. The kernel is now submited

	// wait for all queue submissions to complete
	Q.wait();


  return 0;
}
