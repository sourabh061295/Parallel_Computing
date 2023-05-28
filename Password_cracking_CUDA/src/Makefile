all:cpumd5.c
	gcc -o cpumd5_3letter -O2 -std=c99 cpumd5_3letter.c
	gcc -o cpumd5_4letter -O2 -std=c99 cpumd5_4letter.c
	gcc -o cpumd5_6letter -O2 -std=c99 cpumd5_6letter.c
	nvcc md5_3letter.cu -o md5_3letter
	nvcc md5_4letter.cu -o md5_4letter
	nvcc md5_6letter.cu -o md5_6letter
	
cpu:cpumd5.c
	./cpumd5_3letter
	./cpumd5_4letter
	./cpumd5_6letter

gpu:cpumd5.c
	./md5_3letter
	./md5_4letter
	./md5_6letter
	
clean:
	rm cpumd5_3letter
	rm cpumd5_4letter
	rm cpumd5_6letter
	rm md5_3letter
	rm md5_4letter
	rm md5_6letter

