/*
 * Created on Jan 30, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package transformations;

import fileIO.OutFile;
import java.io.FileReader;
import weka.core.Instance;
import weka.core.Instances;

public class BoxTidwell {
	
	public static Instances transformRegressor(Instances data, int pos,int resultPos, double[] powers)
	{

//1. Get values of the attribute of interest. 
		
//Confusingly, am working with attributes in rows not columns		
		double[] temp=data.attributeToDoubleArray(pos);
		double[] originalData= new double[temp.length];
		double[] logData= new double[temp.length];
		
		for(int i=0;i<temp.length;i++)
		{
			originalData[i]=temp[i];
			logData[i]=Math.log(temp[i]);	
		}
		double[] y =data.attributeToDoubleArray(data.classIndex()); 
//		I'm not sure if this is a memory copy or a reference copy, so be safe
		double[][] transposeFirst = new double[data.numAttributes()][data.numInstances()];
		double[][] transposeSecond = new double[data.numAttributes()+1][data.numInstances()];
		for(int j=0;j<data.numInstances();j++)
		{
			transposeFirst[0][j]=transposeSecond[0][j]=1;
		}
		for(int i=1;i<data.numAttributes();i++)
		{
			transposeFirst[i]=transposeSecond[i]=data.attributeToDoubleArray(i-1);
		}
//		Add one to pos cos of the ones
		pos=pos+1;
//		Second has an attribute at the end of data for transform
		int workingPos=data.numAttributes();
		LinearModel l1,l2;
		double alpha=1, b1,b2;
		double min=0.1;
		boolean finished=false;
		int count=0;
		final int MaxIterations=10;
		//		Initialise alpha to 1
//Find Base SSE		
		//While not termination condition
		while(!finished)
		{
//			System.out.println(" Iteration = "+(count+1)+" alpha = "+alpha);
			//Create new attributes
			//1. Calculate x^alpha
			for(int j=0;j<originalData.length;j++)
			{
				transposeSecond[pos][j]=transposeFirst[pos][j]=Math.pow(originalData[j],alpha);
			}

			//2. Fit y=b1+ .. b_pos	x^alpha (+ other terms)-> get b_pos
			l1=new LinearModel(transposeFirst,y);	
			l1.fitModel();
			
//Not necessary: 
//			l1.formTrainPredictions();
//			l1.findTrainStatistics();
//			System.out.println(l1+"\nVariance for L1 = "+l1.variance);
			
			b1=l1.paras[pos];
			//3. Fit y=b*1+ .. b*_pos	x^alpha +b*_workingPos x^alpha*log(x) (+ other terms)-> get b*2
			//2. Calculate x^alpha*log(x)
			for(int j=0;j<originalData.length;j++)
				transposeSecond[workingPos][j]=transposeFirst[pos][j]*logData[j];
			l2=new LinearModel(transposeSecond,y);	
			l2.fitModel();
			
//			Not necessary: 
//			l2.formTrainPredictions();
//			l2.findTrainStatistics();
//			System.out.println(l2+"\nVariance for L2 = "+l2.variance);
			
			b2=l2.paras[workingPos];
			
			alpha+=b2/b1;
			//Work out change term alpha = b*2/b1+alpha0
//			System.out.println("New Alpha ="+alpha+" b1 = "+b1+" b2 = "+b2);
			//Update termination criteria: stop if small change: check notes
			count++;
			if(Math.abs(b2/b1)<min || count>=MaxIterations)
				finished=true;
			else if(Math.abs(alpha)>10)
			{
				alpha=1;
				finished=true;
			}
		}
//Fix original 
		powers[resultPos]=alpha;
		pos=pos-1;
		Instance inst;
		for(int i=0;i<data.numInstances();i++)
		{
			inst=data.instance(i);
			inst.setValue(pos,Math.pow(originalData[i],alpha));
		}
		return data;
	}

//First rows is all ones
//Last row is for transformed attribute	
	public static double transformRegressor(double[][] data, double[] response, int pos)
	{

//1. Get values of the attribute of interest. 
		double[] temp=data[pos];
		double[] originalData= new double[temp.length];
		double[] transformedData= new double[temp.length];
		double[] logData = new double[originalData.length];
		for(int i=0;i<originalData.length;i++)
		{
			originalData[i]=temp[i];
			logData[i]=Math.log(originalData[i]);	
		}
		double[] y =response; 
		double[][] transposeFirst = new double[data.length][];
		double[][] transposeSecond = new double[data.length+1][];
		for(int j=0;j<data.length;j++)
			transposeFirst[j]=transposeSecond[j]=data[j];
		transposeFirst[pos]=transformedData;
		transposeSecond[pos]=transformedData;		
		transposeSecond[data.length]=logData;
		int workingPos=data.length;
		LinearModel l1,l2;
		double alpha=1, b1,b2;
		double min=0.1;
		boolean finished=false;
		int count=0;
		final int MaxIterations=10;
		//		Initialise alpha to 1
//Find Base SSE		
		//While not termination condition
		while(!finished)
		{
			//Create new attributes
			//1. Calculate x^alpha
			for(int j=0;j<originalData.length;j++)
				transformedData[j]=Math.pow(originalData[j],alpha);

			//2. Fit y=b1+ .. b_pos	x^alpha (+ other terms)-> get b_pos
			l1=new LinearModel(transposeFirst,y);	
			l1.fitModel();
			
//Not necessary: 
//			l1.formTrainPredictions();
//			l1.findTrainStatistics();
//			System.out.println(l1+"\nVariance for L1 = "+l1.variance);
			
			b1=l1.paras[pos];
			//3. Fit y=b*1+ .. b*_pos	x^alpha +b*_workingPos x^alpha*log(x) (+ other terms)-> get b*2
			//2. Calculate x^alpha*log(x)
			for(int j=0;j<originalData.length;j++)
				transposeSecond[workingPos][j]=originalData[j]*logData[j];
			l2=new LinearModel(transposeSecond,y);	
			l2.fitModel();
			
			b2=l2.paras[workingPos];
			
			alpha+=b2/b1;
			//Work out change term alpha = b*2/b1+alpha0
			//Update termination criteria: stop if small change: check notes
			count++;
			if(Math.abs(b2/b1)<min || count>=MaxIterations)
				finished=true;
			else if(Math.abs(alpha)>10)
			{
				alpha=1;
				finished=true;
			}
		}
//Fix original 
		return alpha;
	}
	
	public static void main(String[] args)
	{
		Instances data=null;
		try{
			FileReader r = new FileReader("C:/Research/Code/Archive Generator/src/weka/addOns/BoxTidwellTest2.arff");
			data = new Instances(r);
			data.setClassIndex(data.numAttributes()-1);
		}catch(Exception e)
		{
			System.out.println("Error loading file "+e);
		}
		double[] powers=new double[data.numAttributes()-1];
//		data=transformRegressor(data,0,powers);
//		data=transformRegressor(data,2,powers);
//		data=transformRegressor(data,1,powers);

		System.out.println(" Final powers =");
		for(int i=0;i<powers.length;i++)
			System.out.println(i+" ="+powers[i]);
			
		OutFile r = new OutFile("C:/Research/Code/Archive Generator/src/weka/addOns/BoxTidwellResults2.arff");
		r.writeLine(data.toString());
		
	}
	
}
