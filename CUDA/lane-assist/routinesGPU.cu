#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"
#include "png_io.h"


void canny(uint8_t *img, uint8_t *image_out, int height, int width, float level);
void houghTransform();
void getLines();

__global__ void NoiseReduct(uint8_t *im, float *NR, int height, int width);
__global__ void IntensityGrad(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width);
__global__ void Edge(uint8_t *pedge, float *G, float *phi, int height, int width);
__global__ void Umbral(uint8_t *pedge, uint8_t *image_out, float *G, int height, int width, float level);

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{

	/*

	TODO

	Book mem and copy Host to Device

	void canny
	view BWfile
	void houghTransform
	void getLines

	Copy Device to Host and free mem

	*/

	//Canny
	uint8_t *img, *imEdge;
	int size=height*width*sizeof(uint8_t);
	cudaMalloc((void**)&img,size);
	cudaMemcpy(img, im, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&imEdge,size);

	canny(img, imEdge, height,width, 1000.0f);
	cudaFree(img);//Reorganizacion admin mem glob gpu

	write_png_fileBW("out_edges.png", imEdge,width,height);


	//Free mem (Posible reorganizacion para evitar sobrecargar la mem global de GPU)
	//cudaFree(img);
	cudaFree(imEdge);
}



void canny(uint8_t *img, uint8_t *image_out, int height, int width, float level) 
{

	/*

	Noise reduction Kernel?)
	Intensity gradient Kernel?)
	Edge Kernel?)
	Hysteresis Thresholding kernel?)

	*/
	dim3 dimGrid(1,1);
	dim3 dimBlock(1,1);

	float *NR;
	int size = height * width *sizeof(float);
	cudaMalloc((void**)&NR, size);


	NoiseReduct<<<dimGrid, dimBlock>>>(img, NR, height, width);

	float *G, *Gx, *Gy, *phi;
	cudaMalloc((void**)&G,size);
	cudaMalloc((void**)&Gx,size);
	cudaMalloc((void**)&Gy,size);
	cudaMalloc((void**)&phi,size);

	IntensityGrad<<<dimGrid,dimBlock>>>(NR ,Gx, Gy, G, phi, height, width);

	cudaFree(NR);
	cudaFree(Gx);
	cudaFree(Gy);

	uint8_t *pedge;
	cudaMalloc((void**)&pedge, size);

	Edge<<<dimGrid,dimBlock>>>(pedge, G, phi, height, width);

	cudaFree(phi);

	Umbral<<<dimGrid,dimBlock>>>(pedge, image_out, G, height, width, level);

	cudaFree(pedge);
	cudaFree(G);

	//Free mem
	//cudaFree(NR);
	//cudaFree(NR);
	//cudaFree(Gx);
	//cudaFree(Gy);
	//cudaFree(G);
	//cudaFree(phi);
	//cudaFree(pedge);
	//cudaFree(G);
}

__global__ void NoiseReduct(uint8_t *im, float *NR, int height, int width){
	int i ,j;
	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			
			NR[i*width+j] =
				(2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}

}

__global__ void IntensityGrad(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width){
	int i, j;
	float PI=3.141593;
	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			
			Gx[i*width+j] = 
				(1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy[i*width+j] = 
				((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
	}

}

__global__ void Edge(uint8_t *pedge, float *G, float *phi, int height, int width){
	int i, j;
	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
		}

}

__global__ void Umbral(uint8_t *pedge, uint8_t *image_out, float *G, int height, int width, float level){
	float lowthres = level/2;
	float hithres  = 2*(level);
	int i, j, ii, jj;

	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
		}
}

void houghTransform(){
	//Could be a kernerl by selfs
}

void getLines (){
	//Could be a kernerl by selfs
	
}

