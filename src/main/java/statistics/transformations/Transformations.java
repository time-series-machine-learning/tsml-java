package transformations;


import statistics.distributions.NormalDistribution;
import weka.core.*;

// Null lass does not change anything

abstract public class Transformations {
//Transform data dependent on the 
	boolean supervised=false;
	boolean response=false;
	public boolean isSupervised(){return supervised;}
	public boolean isResponseTransform(){return response;}
	abstract public Instances transform(Instances data);
	abstract public Instances invert(Instances data);
//Transform data based on values formed by calling transform on another data set
//Only needed for dependent variable transformations, for others does nothing
	abstract public Instances staticTransform(Instances data);
	abstract public double[] invertPredictedResponse(double[] d);
	public double[] invertPredictedQuantiles(double[] d)
	{
		return invertPredictedResponse(d);
	}
	public static int width=100;
	static public void setWidth(int w){width=w;}
	static public double[] getNormalQuantiles(double mean,double variance)
	{
		double[] q= new double[width];
		NormalDistribution norm = new NormalDistribution(mean,Math.sqrt(variance));
		
		for(int i=0;i<width;i++)
			q[i]=norm.getQuantile((i+1)/(double)width);
		return q;
	}
	static public double[] getNormalQuantiles(double[] standard, double mean,double variance)
	{
		double stDev=Math.sqrt(variance);
		double[] q= new double[standard.length];
		for(int i=0;i<standard.length;i++)
			q[i]=standard[i]*stDev+mean;
		return q;
	}
	public double[] findResiduals(double[] actual, double[] fitted)
	{
		if(actual.length!=fitted.length)
		{
			System.out.println(" Error, mismatched lengths in findResiduals");
			System.exit(0);
		}
		double[] res= new double[actual.length];
		for(int i=0;i<res.length;i++)
			res[i]=actual[i]-fitted[i]; 
		return res;
	}

	static public Instances powerTransform(Instances data, int[] pos, double[] powers)
	{
		Instance inst;
		int p=data.classIndex();
        for(int i=0;i<data.numInstances();i++)
		{
        	inst=data.instance(i);
        	for(int j=0;j<pos.length;j++)
        	{
        		if(pos[j]!=p)
        		{
        			if(powers[j]!=0)
        				inst.setValue(pos[j], Math.pow(inst.value(j),powers[j]));
        			else
        				inst.setValue(pos[j], Math.log(inst.value(j)));
        		}
        	}
		}
        return data;
	}
	
	
}
