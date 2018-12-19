/*
 * Created on Jan 29, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package transformations;

import fileIO.*;
import weka.core.*;
import weka.classifiers.functions.*;
import weka.classifiers.*;
/**
 * @author ajb
 *
 * TODO To chang
 * 
 * e the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class BoxCox extends Transformations{

	public static double MIN=-3,MAX=3,INTERVAL=0.25;
	boolean tryZero=false;
	AbstractClassifier c;
	double minError=Double.MAX_VALUE, bestLambda;
	double gamma;
	boolean strictlyPositive=true;

	public BoxCox()
	{
		supervised=true;
		response=true;
		minError=Double.MAX_VALUE;
		bestLambda=MIN;
		c=new LinearRegression();
		String[] options = {"-S 1","-C "};
		try{
			c.setOptions(options);
		}catch(Exception e){
			System.out.println(" Error Setting options in constructor");
		}
	}
	public void setStrictlyPos(boolean f){strictlyPositive=f;}
	public BoxCox(AbstractClassifier c)
	{
		this();
		this.c=c;
	}
//Performs a specific B-C transform on the response variable, overwriting original
	static public void transformResponse(Instances data, double lambda, double[] response)
	{
		Instance inst;
		double v;
		int responsePos=data.numAttributes()-1;
		for(int i=0;i<response.length;i++)
		{
			inst=data.instance(i);
			
			v=(Math.pow(response[i],lambda)-1)/lambda;
			inst.setValue(responsePos,v);
		}
		
	}
	
	
	
//	Transform the response variable using box-cox procedure
	public Instances transform(Instances data)
	{
		int responsePos=data.classIndex();
		double[] response=data.attributeToDoubleArray(responsePos);
		double[] predictions=new double[response.length];	
		double v;
		Instance inst;
//Check if strictly positive
		gamma=response[0];
		for(int i=1;i<response.length;i++)
		{
			if(response[i]<gamma)
				gamma=response[i];
		}
		System.out.println(" Min value = "+gamma);
		if(gamma<=0)
		{
			gamma=-2*gamma+1;
			System.out.println(" Data series is not strictly positive, rescaling by "+gamma);
			for(int i=0;i<response.length;i++)
				response[i]+=gamma;
		}	
		
		for(double lambda=MIN;lambda<=MAX;lambda+=INTERVAL)
		{
//Transform response			
			if(lambda==0) lambda+=INTERVAL;
			transformResponse(data,lambda,response);

	//Fit model and get training predictions
			try{
				c.buildClassifier(data);
//				System.out.println("Classifier = "+c);
				
				
				for(int i=0;i<predictions.length;i++)
				{
					inst=data.instance(i);
					predictions[i]=c.classifyInstance(inst);
//					if(predictions[i]<0)
//						predictions[i]=0;
				}
			}
			catch(Exception e)
			{
				System.out.println(" Error building with lambda = "+lambda);
			}
//Assess quality of fit by SSE: Transformed or untransformed? Assume we have to 
// turn it back
			
			double SSE=0;
			boolean f=true;
			for(int i=0;i<predictions.length;i++)
			{
				predictions[i]*=lambda;
				predictions[i]++;
				if(predictions[i]<=0)
					predictions[i]=0;
				else
				{
					if(lambda>0)
						predictions[i]=Math.pow(predictions[i],1.0/lambda);
					else
						predictions[i]=1/Math.pow(predictions[i],-1.0/lambda);
				}	
				SSE+=(predictions[i]-response[i])*(predictions[i]-response[i]);
			}
//Check whether minimum, and store
			SSE/=(data.numInstances()-data.numAttributes());
			System.out.println("lambda = "+lambda+"SSE ="+SSE);
			if(SSE<minError)
			{
				minError=SSE;
				bestLambda=lambda;
			}
		}
		System.out.println("Min lambda = "+bestLambda+" with MSE = "+minError);
//Perform best transform
		for(int i=0;i<response.length;i++)
		{
			inst=data.instance(i);
			v=(Math.pow(response[i],bestLambda)-1)/bestLambda;
			inst.setValue(responsePos,v);
		}
		return data;
		
	}
	public Instances invert(Instances data){
		Instance inst;
		int responsePos=data.numAttributes()-1;
		double[] response=data.attributeToDoubleArray(responsePos);
		double v;
		
		for(int i=0;i<data.numInstances();i++)
		{
			inst=data.instance(i);
			v=response[i]*bestLambda;
			v++;
			v=Math.pow(v,1/bestLambda);			
			inst.setValue(responsePos,v);
		}
		return data;
	}
//Transform data based on values formed by calling transform on another data set
//Only needed for dependent variable transformations, for others does nothing
	public Instances staticTransform(Instances data)
	{
		Instance inst;
		int responsePos=data.numAttributes()-1;
		double[] response=data.attributeToDoubleArray(responsePos);
		double v;
		
		for(int i=0;i<data.numInstances();i++)
		{
			inst=data.instance(i);
			v=(Math.pow(response[i],bestLambda)-1)/bestLambda;
			inst.setValue(responsePos,v);
		}
		return data;
	}
	public double[] invertPredictedResponse(double[] d)
	{
		double v;
		for(int i=0;i<d.length;i++)
		{
			v=d[i]*bestLambda;
			v++;
			d[i]=Math.pow(v,1/bestLambda);			
		}
		return d;
	}
	public static void main(String[] args)
	{
		double[] quantiles = Transformations.getNormalQuantiles(0.0,1.0);
		for(int i=0;i<quantiles.length;i++)
			System.out.println("Quantile "+i+" = "+quantiles[i]);
		OutFile of = new OutFile("TestQuantiles.csv");
		for(int i=0;i<quantiles.length;i++)
		{
			System.out.println(i+","+(i+1)/(double)quantiles.length+","+quantiles[i]);
			of.writeLine(i+","+(i+1)/(double)quantiles.length+","+quantiles[i]);
		}
	}

}
