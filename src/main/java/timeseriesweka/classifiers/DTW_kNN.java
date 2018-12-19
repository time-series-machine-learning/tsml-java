package timeseriesweka.classifiers;
import java.io.FileReader;
import weka.classifiers.lazy.kNN;

import weka.core.*;

import weka.core.EuclideanDistance;
import timeseriesweka.elastic_distance_measures.DTW;

/* This class is a specialisation of kNN that can only be used with the efficient DTW distance
 * 
 * The reason for specialising is this class has the option of searching for the optimal window length
 * through a grid search of values.
 * 
 * By default this class does a search. 
 * To search for the window size call
 * optimiseWindow(true);
 * By default, this does a leave one out cross validation on every possible window size, then sets the 
 * proportion to the one with the largest accuracy. This will be slow. Speed it up by
 * 
 * 1. Set the max window size to consider by calling
 * setMaxWindowSize(double r) where r is on range 0..1, with 1 being a full warp.
 * 
 * 2. Set the increment size 
 * setIncrementSize(int s) where s is on range 1...trainSetSize 
 * 
 * This is a basic brute force implementation, not optimised! There are probably ways of 
 * incrementally doing this. It could further be speeded up by using PAA to reduce the dimensionality first.
 * 
 */

public class DTW_kNN extends kNN {
	private boolean optimiseWindow=false;
	private double windowSize=0.1;
	private double maxWindowSize=1;
	private int incrementSize=10;
	private Instances train;
	private int trainSize;
	private int bestWarp;
	DTW dtw=new DTW();
	
//	DTW_DistanceEfficient dtw=new DTW_DistanceEfficient();
	public DTW_kNN(){
		super();
		dtw.setR(windowSize);
		setDistanceFunction(dtw);
		super.setKNN(1);
	}
	
	public void optimiseWindow(boolean b){ optimiseWindow=b;}
	public void setMaxR(double r){ maxWindowSize=r;}
	
	
	public DTW_kNN(int k){
		super(k);
		dtw.setR(windowSize);
		optimiseWindow=true;
		setDistanceFunction(dtw);
	}
	public void buildClassifier(Instances d){
		dist.setInstances(d);
		train=d;
		trainSize=d.numInstances();
		if(optimiseWindow){
			

			double maxR=0;
			double maxAcc=0;
/*Set the maximum warping window: Not this is all a bit mixed up. 
The window size in the r value is range 0..1, but the increments should be set by the 
data*/
			int dataLength=train.numAttributes()-1;
			int max=(int)(dataLength*maxWindowSize);
//			System.out.println(" MAX ="+max+" increment size ="+incrementSize);
			for(double i=0;i<max;i+=incrementSize){
				//Set r for current value
				dtw.setR(i/(double)dataLength);
				double acc=crossValidateAccuracy();
//				System.out.println("\ti="+i+" r="+(i/(double)dataLength)+" Acc = "+acc);
				if(acc>maxAcc){
					maxR=i/dataLength;
					maxAcc=acc;
//					System.out.println(" Best so far ="+maxR +" Warps ="+i+" has Accuracy ="+maxAcc);
				}
			}
			bestWarp=(int)(maxR*dataLength);
			dtw.setR(maxR);
//			System.out.println(" Best R = "+maxR+" Best Warp ="+bestWarp+" Size = "+(maxR*dataLength));
		}
// Then just use the normal kNN with the DTW distance. Not doing this any more because its slow!
		super.buildClassifier(d);
	}
/* No need to do this, since we can use the IBk version, which should be optimised!
	public double classifyInstance(Instance d){
//Basic distance, with early abandon, which has not been implemented in the distance comparison.		This is only for nearest neighbour
		double minSoFar=Double.MAX_VALUE;
		double dist; int index=0;
		for(int i=0;i<train.numInstances();i++){
			dist=dtw.distance(train.instance(i),d,minSoFar);
			if(dist<minSoFar){
				minSoFar=dist;
				index=i;
			}
		}
		return train.instance(index).classValue();
	}
*/
//Could do this for BER instead	
	private double crossValidateAccuracy(){
		double a=0,d=0, minDist;
		int nearest=0;
		Instance inst;
		for(int i=0;i<trainSize;i++){
//Find nearest to element i
			nearest=0;
			minDist=Double.MAX_VALUE;
			inst=train.instance(i);
			for(int j=0;j<trainSize;j++){
				if(i!=j){
//					if(i==0&&j<2)
//						System.out.println("\t"+inst+" and \n\t"+train.instance(j)+"\n\t\t ="+d);
					d=dtw.distance(inst,train.instance(j),minDist);
					if(d<minDist){
						nearest=j;
						minDist=d;
					}
				}
			}
//			System.out.println("\t\tDistance between "+i+" and "+nearest+" ="+minDist);
			
			//Measure accuracy for nearest to element i			
			if(inst.classValue()==train.instance(nearest).classValue())
				a++;
		}
		return a/(double)trainSize;
	}
	
	
	public static void main(String[] args){
		DTW_kNN c = new DTW_kNN();
		String path="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

		Instances test=loadData(path+"Coffee\\Coffee_TEST.arff");
		Instances train=loadData(path+"Coffee\\Coffee_TRAIN.arff");
		train.setClassIndex(train.numAttributes()-1);
		c.buildClassifier(train);
		
	}
	public static Instances loadData(String fileName)
	{
		Instances data=null;
		try{
			FileReader r;
			r= new FileReader(fileName); 
			data = new Instances(r); 

			data.setClassIndex(data.numAttributes()-1);
		}catch(Exception e)
		{
			System.out.println(" Error ="+e+" in method loadData");
		}
		return data;
	}

}
