#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"
#include "png_io.h"


// const int BLOCK_SIZE = 1;
// const int GRID_SIZE = 1;

const int BLOCK_SIZE = 16;

void canny(uint8_t *img, uint8_t *image_out, int height, int width, float level);


__global__ void NoiseReduct(uint8_t *im, float *NR, int height, int width);
__global__ void IntensityGrad(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width);
__global__ void Edge(uint8_t *pedge, float *G, float *phi, int height, int width);
__global__ void Umbral(uint8_t *pedge, uint8_t *image_out, float *G, int height, int width, float level);
__global__ void houghTransform(uint8_t *im, int height, int width, uint32_t *accumulators, int accu_width, int accu_height, float *sin_table, float *cos_table, float hough_h);
__global__ void getLines(uint32_t *accumulators, int accu_width, int accu_height, int height, int width,int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines, float *sin_table, float *cos_table);

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines, float *sin_table, float *cos_table)
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

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	uint32_t *accumulators;
	int accu_width = 180;
	int accu_height = hough_h * 2.0;
	cudaMalloc((void**)&accumulators, accu_width*accu_height*sizeof(uint32_t));
	
	float *sin_table_d, *cos_table_d;
	cudaMalloc((void**)&sin_table_d, 180*sizeof(float));
	cudaMalloc((void**)&cos_table_d, 180*sizeof(float));
	cudaMemcpy(sin_table_d, sin_table, 180*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cos_table_d, cos_table, 180*sizeof(float), cudaMemcpyHostToDevice);

	int grid_size=sizeof(float)*height/BLOCK_SIZE;

	dim3 dimGrid((sizeof(float)*height/BLOCK_SIZE)+1,(sizeof(float)*width/BLOCK_SIZE)+1);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	houghTransform<<<dimGrid, dimBlock>>>(imEdge, height, width, accumulators, accu_width, accu_height, sin_table_d, cos_table_d, hough_h);

	cudaFree(imEdge);//Reorganizacion admin mem glob gpu

	int *x1_d, *y1_d, *x2_d, *y2_d;
	cudaMalloc((void**)&x1_d, 10*sizeof(int));
	cudaMalloc((void**)&x2_d, 10*sizeof(int));
	cudaMalloc((void**)&y1_d, 10*sizeof(int));
	cudaMalloc((void**)&y2_d, 10*sizeof(int));

	int *lines_d;
	cudaMalloc((void**)&lines_d, sizeof(int));
	cudaMemcpy(lines_d, lines, sizeof(int), cudaMemcpyHostToDevice);

	getLines<<<dimGrid, dimBlock>>>(accumulators, accu_width, accu_height, height, width, x1_d, y1_d, x2_d, y2_d, lines_d, sin_table_d, cos_table_d);

	cudaMemcpy(nlines, lines_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x1, x1_d, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2, x2_d, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, y1_d, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2, y2_d, 10*sizeof(int), cudaMemcpyDeviceToHost);

	//Free mem (Posible reorganizacion para evitar sobrecargar la mem global de GPU)
	//cudaFree(img);
	//cudaFree(imEdge)
	cudaFree(accumulators);
	cudaFree(sin_table_d);
	cudaFree(cos_table_d);
	cudaFree(x1_d);
	cudaFree(x2_d);
	cudaFree(y1_d);
	cudaFree(y2_d);
	cudaFree(lines_d);
}



void canny(uint8_t *img, uint8_t *image_out, int height, int width, float level) {

	/*

	Noise reduction Kernel?)
	Intensity gradient Kernel?)
	Edge Kernel?)
	Hysteresis Thresholding kernel?)

	*/
dim3 dimGrid((sizeof(float)*height/BLOCK_SIZE)+1,(sizeof(float)*width/BLOCK_SIZE)+1);
dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);

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

	int j=blockidx.x*blockDim.x+threadIdx.x;
	int i=blockidx.y*blockDim.y+threadIdx.y;
	if(i>2 && i<height-2 && j>2 &&j<width-2){

		NR[i*width+j] =(2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
		+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
		+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
		+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
		+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
		/159.0;

	}

}

__global__ void IntensityGrad(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width){

	float PI=3.141593;
	int j=blockidx.x*blockDim.x+threadIdx.x;
	int i=blockidx.y*blockDim.y+threadIdx.y;
	if(i>2 && i<height-2 && j>2 &&j<width-2){
		
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

	int j=blockidx.x*blockDim.x+threadIdx.x;
	int i=blockidx.y*blockDim.y+threadIdx.y;
	if(i>3 && i<height-3 && j>3 &&j<width-3){
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
	// float lowthres = level/2;
	// float hithres  = 2*(level);
	// int i, j, ii, jj;

	// for(i=3; i<height-3; i++)
	// 	for(j=3; j<width-3; j++)
	// 	{
	// 		image_out[i*width+j] = 0;
	// 		if(G[i*width+j]>hithres && pedge[i*width+j])
	// 			image_out[i*width+j] = 255;
	// 		else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
	// 			// check neighbours 3x3
	// 			for (ii=-1;ii<=1; ii++)
	// 				for (jj=-1;jj<=1; jj++)
	// 					if (G[(i+ii)*width+j+jj]>hithres)
	// 						image_out[i*width+j] = 255;
	// 	}
	

	int ii, jj;
	float lowthres = level/2;
	float hithres  = 2*(level);
	int j=blockidx.x*blockDim.x+threadIdx.x;
	int i=blockidx.y*blockDim.y+threadIdx.y;
	if(i>3 && i<height-3 && j>3 &&j<width-3){

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

__global__ void houghTransform(uint8_t *im, int height, int width, uint32_t *accumulators, int accu_width, int accu_height, float *sin_table, float *cos_table, float hough_h){
	//Could be a kernerl by it selfs
	int i, j, theta;

	for(i=0; i<accu_width*accu_height; i++)
		accumulators[i]=0;	

	float center_x = width/2.0; 
	float center_y = height/2.0;
	for(i=0;i<height;i++)  
	{  
		for(j=0;j<width;j++)  
		{  
			if( im[ (i*width) + j] > 250 ) // Pixel is edge  
			{  
				for(theta=0;theta<180;theta++)  
				{  
					float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]++;

				} 
			} 
		} 
	}
}

__global__ void getLines(uint32_t *accumulators, int accu_width, int accu_height, int height, int width,int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines, float *sin_table, float *cos_table){
	//Could be a kernerl by it selfs
	int threshold;
	if (width > height)
	{
		threshold = width / 6;
	}
	else
	{
		threshold = height / 6;
	}

	int rho, theta, ii, jj;
	uint32_t max;

	for (rho = 0; rho < accu_height; rho++)
	{
		for (theta = 0; theta < accu_width; theta++)
		{

			if (accumulators[(rho * accu_width) + theta] >= threshold)
			{
				// Is this point a local maxima (9x9)
				max = accumulators[(rho * accu_width) + theta];
				for (int ii = -4; ii <= 4; ii++)
				{
					for (int jj = -4; jj <= 4; jj++)
					{
						if ((ii + rho >= 0 && ii + rho < accu_height) && (jj + theta >= 0 && jj + theta < accu_width))
						{
							if (accumulators[((rho + ii) * accu_width) + (theta + jj)] > max)
							{
								max = accumulators[((rho + ii) * accu_width) + (theta + jj)];
							}
						}
					}
				}

				if (max == accumulators[(rho * accu_width) + theta]) // local maxima
				{
					int x1, y1, x2, y2;
					x1 = y1 = x2 = y2 = 0;

					if (theta >= 45 && theta <= 135)
					{
						if (theta > 90)
						{
							// y = (r - x cos(t)) / sin(t)
							x1 = width / 2;
							y1 = ((float)(rho - (accu_height / 2)) - ((x1 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;
							y2 = ((float)(rho - (accu_height / 2)) - ((x2 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
						}
						else
						{
							// y = (r - x cos(t)) / sin(t)
							x1 = 0;
							y1 = ((float)(rho - (accu_height / 2)) - ((x1 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width * 2 / 5;
							y2 = ((float)(rho - (accu_height / 2)) - ((x2 - (width / 2)) * cos_table[theta])) / sin_table[theta] + (height / 2);
						}
					}
					else
					{
						// x = (r - y sin(t)) / cos(t);
						y1 = 0;
						x1 = ((float)(rho - (accu_height / 2)) - ((y1 - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
						y2 = height;
						x2 = ((float)(rho - (accu_height / 2)) - ((y2 - (height / 2)) * sin_table[theta])) / cos_table[theta] + (width / 2);
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}

