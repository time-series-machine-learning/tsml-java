package distance.elastic;

import java.util.concurrent.ConcurrentLinkedQueue;

public class MemorySpaceProvider {
	private static MemorySpaceProvider singleton = null;
	
	ConcurrentLinkedQueue<double[][]>dblMatrices;
	ConcurrentLinkedQueue<int[][]>intMatrices;
	ConcurrentLinkedQueue<double[]>dblArrays;
	
	int size;
	
	
	private MemorySpaceProvider(int length){
		int nThreads = Runtime.getRuntime().availableProcessors();
		this.size = length+10;//+10 to be sure we have enough
		dblMatrices = new ConcurrentLinkedQueue<>();
		intMatrices = new ConcurrentLinkedQueue<>();
		dblArrays = new ConcurrentLinkedQueue<>();
//		System.out.println("Creating memory space provider with size="+size);
		for (int i = 0; i < 2*nThreads;i++) {
			dblMatrices.add(new double[size][size]);
			intMatrices.add(new int[size][size]);
		}
		for (int i = 0; i < 10*nThreads; i++) {
			dblArrays.add(new double[size]);
		}
		
	}

	public static MemorySpaceProvider getInstance(int length) {
		if (singleton == null || length>singleton.size) {
			synchronized (MemorySpaceProvider.class) {
				if (singleton == null|| length>singleton.size) {
					singleton = new MemorySpaceProvider(length);
				}
			}
		}
		return singleton;
	}
	
	public static MemorySpaceProvider getInstance() {
		if (singleton == null) {
			synchronized (MemorySpaceProvider.class) {
				if (singleton == null) {
					singleton = new MemorySpaceProvider(3000);
				}
			}
		}
		return singleton;
	}
	
	public double[][] getDoubleMatrix(){
		return dblMatrices.poll();
	}
	
	public void returnDoubleMatrix(double[][]m){
		dblMatrices.add(m);
	}
	
	public int[][] getIntMatrix(){
		return intMatrices.poll();
	}
	
	public void returnIntMatrix(int[][]m){
		intMatrices.add(m);
	}
	
	public double[] getDoubleArray(){
		return dblArrays.poll();
	}
	
	public void returnDoubleArray(double[]t){
		dblArrays.add(t);
	}

}