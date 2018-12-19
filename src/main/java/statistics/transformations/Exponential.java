package transformations;

import weka.core.*;

/* Implements a simple log transformation of the response variable
 * 
 * 
 */
public class Exponential extends Transformations{
	double offSet=0;
	static double zeroOffset=1;
	public Exponential()
	{
		supervised=true;
		response=true;
	}
	public Instances transform(Instances data){
//Not ideal, should call a method to get this
		int responsePos=data.numAttributes()-1;
		System.out.println(" Response Pos = "+responsePos);
		double[] response=data.attributeToDoubleArray(responsePos);
//Find the min value
		double min=response[0];
		for(int i=0;i<response.length;i++)
		{
			if(response[i]<min)
				min=response[i];
		}
		if(min<=zeroOffset)	//Cant take a log of a negative, so offset
		{
			offSet=-min+zeroOffset;
		}
		else
			offSet=0;
		System.out.println(" Min value = "+min+" offset = "+offSet);
		
		for(int i=0;i<data.numInstances();i++)
		{
			Instance t = data.instance(i);
			double resp=t.value(responsePos);
			System.out.print(i+" "+resp);
			resp=Math.log(resp+offSet);
			System.out.println(" "+resp);
			t.setValue(responsePos,resp);
		}
		return data;
	}
	public Instances invert(Instances data){
		int responsePos=data.numAttributes()-1;
		for(int i=0;i<data.numInstances();i++)
		{
			Instance t = data.instance(i);
			double resp=t.value(responsePos);
			resp=Math.exp(resp);
			resp-=offSet;
			t.setValue(responsePos,resp);
		}
		return data;
		
	}
	public double[] invertPredictedResponse(double[] d){
		for(int i=0;i<d.length;i++)
		{
			d[i]=Math.exp(d[i]);
			d[i]-=offSet;
		}
		return d;
		
	}
	//Get quantile values for a transformed response, assuming
//Mean and variance of model	
	
	
	//Not relevant, only needed for st
	public Instances staticTransform(Instances data){
		return data;
	}

}
