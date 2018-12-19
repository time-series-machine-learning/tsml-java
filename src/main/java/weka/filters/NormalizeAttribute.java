/** Class NormalizeAttribute.java
 * 
 * @author AJB
 * @version 1
 * @since 14/4/09
 * 
 * Class normalizes attributes, basic version. 
 		1. Assumes no missing values. 
 		2. Assumes all attributes real values
 		3. Assumes class index same in all data (vague checks made) but can be none set (classIndex==-1)
 		4. Batch process, by default it calculates the ranges from the instances in trainData, then uses
 		this to process the instances passed. Note that this may produce values outside the 
 		interval range, since the min or max of the test data may be separate. If you want to avoid
 		this, the only way at the moment is to first merge train and test, then pass the merged set.
 		Easy to hack round this if I have to.
 
 * Normalise onto [0,1] if norm==NormType.INTERVAL, 
 * Normalise onto Normal(0,1) if norm==NormType.STD_NORMAL, 
 * 
 * Useage:
 * 	Instances train = //Get Train
 * 	Instances test = //Get Train
 * 
 * NormalizeAttributes na = new NormalizeAttributes(train);
 *
 * na.setNormMethod(NormalizeAttribute.NormType.INTERVAL); //Defaults to interval anyway
 	try{
 //Both	processed with the stats from train.
 		Instances newTrain=na.process(train);
 		Instances newTest=na.process(test);
 		
 */

package weka.filters;

import weka.core.Instances;

public class NormalizeAttribute extends SimpleBatchFilter{
	enum NormType {INTERVAL,STD_NORMAL};
	Instances trainData;
	double[] min;
	double[] max;
	double[] mean;
	double[] stdev;
	int classIndex;
	NormType norm=NormType.INTERVAL;
/* 
 * 
 */
	public NormalizeAttribute(Instances data){
		trainData=data;
		classIndex=data.classIndex();
//Finds all the stats, doesnt cost much more really		
		findStats(data);
	}
	protected void findStats(Instances r){
//Find min and max	
//		assert(classIndex==r.classIndex());
		
		max=new double[r.numAttributes()];
		min=new double[r.numAttributes()];
		for(int j=0;j<r.numAttributes();j++)
		{
			max[j]=Double.MIN_VALUE;
			min[j]=Double.MAX_VALUE;
			for(int i=0;i<r.numInstances();i++){
				double x=r.instance(i).value(j);
				if(x>max[j])
					max[j]=x;
				if(x<min[j])
					min[j]=x;
			}
		}
		
//Find mean and stdev		
		mean=new double[r.numAttributes()];
		stdev=new double[r.numAttributes()];
		double sum,sumSq,x,y;
		for(int j=0;j<r.numAttributes();j++)
		{
			sum=0;
			sumSq=0;
			for(int i=0;i<r.numInstances();i++){
				x=r.instance(i).value(j);
				sum+=x;
				sumSq+=x*x;
			}
			stdev[j]=sumSq/r.numInstances()-sum*sum;
			mean[j]=sum/r.numInstances();
			stdev[j]=Math.sqrt(stdev[j]);
		}
	}
	public double[] getRanges(){
		double[] r= new double[max.length];
		for(int i=0;i<r.length;i++)
			r[i]=max[i]-min[i];
		return r;
	}
	//This should probably be connected to trainData?	
	protected Instances determineOutputFormat(Instances inputFormat){
		return new Instances(inputFormat, 0);
	}
	public void setTrainData(Instances data){ //Same as the constructor
		trainData=data;
		classIndex=data.classIndex();
//Finds all the stats, doesnt cost much more really		
		findStats(data);
	}
	public void setNormMethod(NormType n){
		norm=n;
	}
	public Instances process(Instances inst) throws Exception {
//Clones the data. Presupposes find stats has been called! 		
		if(classIndex!=inst.classIndex())
			throw new Exception("Wrong class index ="+inst.classIndex()+" expecting ="+classIndex);

		  Instances result = new Instances(inst);
		  switch(norm){
		  case INTERVAL:
			  intervalNorm(result);
			  break;
		  case STD_NORMAL:
			  standardNorm(result);
			  break;
			  default:
				  System.out.println(" Unknown norm!"+norm);
				 throw new Exception("in process"); 
		  }
		  return result;
	}
/* Wont normalise the class value*/	
	public void intervalNorm(Instances r){
		for(int i=0;i<r.numInstances();i++){
			for(int j=0;j<r.numAttributes();j++){
				if(j!=classIndex){
					double x=r.instance(i).value(j);
					r.instance(i).setValue(j,(x-min[j])/(max[j]-min[j]));
//					System.out.println("instance ="+i+" Attribute ="+j+" Value = "+x+" Min ="+min[j]+" max = "+max[j]);
				}
			}
		}
	}
	
	
	public void standardNorm(Instances r){
		for(int j=0;j<r.numAttributes();j++){
			if(j!=classIndex){
				for(int i=0;i<r.numInstances();i++){
					double x=r.instance(i).value(j);
					r.instance(i).setValue(i,(x-mean[j])/(stdev[j]));
				}
			}
		}
	}
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}


	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
/* Test Harness.
 * 
	public static void main(String[] args){
		Instances test=weka.classifiers.evaluation.ClassifierTools.loadData("C:\\Research\\Data\\WekaTest\\NormalizeTest");
		Instances train=weka.classifiers.evaluation.ClassifierTools.loadData("C:\\Research\\Data\\WekaTest\\NormalizeTrain");
		test.setClassIndex(test.numAttributes()-1);
		train.setClassIndex(test.numAttributes()-1);
		
		NormalizeAttribute na=new NormalizeAttribute(test);
		try{

			na.setNormMethod(NormalizeAttribute.NormType.INTERVAL); //Defaults to interval anyway
			Instances newTrain=na.process(train);
			Instances newTest=na.process(test);
			System.out.println(" Fixed interval train ="+newTrain);
			System.out.println(" Fixed interval test ="+newTest);
			na.setNormMethod(NormalizeAttribute.NormType.STD_NORMAL); //Defaults to interval anyway
			na.setTrainData(train);
			newTrain=na.process(train);
			newTest=na.process(test);
			System.out.println(" Std Normal train ="+newTrain);
			System.out.println(" Std Normal test ="+newTest);
			
		}catch(Exception e){
			System.out.println(" Exception thrown somewhere, caught main ="+e);
		}
		
		
	}
 */
	
	
}
