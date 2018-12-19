/*
 * Created on Jan 29, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package transformations;

import fileIO.OutFile;
import statistics.tests.ResidualTests;
import weka.core.Instance;
import weka.core.Instances;
/**
 * @author ajb
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class YeoJohnson extends BoxCox{

	public YeoJohnson(double r)
	{
		super();
		bestLambda=r;
	}

//Performs a specific B-C transform on the response variable, overwriting original

    @Override
	public Instances invert(Instances data){
		Instance inst;
		int responsePos=data.numAttributes()-1;
		double[] response=data.attributeToDoubleArray(responsePos);
		double v;
		double[] newVals=invert(bestLambda,response);
		
		for(int i=0;i<data.numInstances();i++)
		{
			inst=data.instance(i);
			inst.setValue(responsePos,newVals[i]);
		}
		return data;
	}
    @Override
	public double[] invertPredictedResponse(double[] d)
	{
		return invert(bestLambda,d);
	}
	static public Instances invertResponse(Instances data, double lambda){
		Instance inst;
		int responsePos=data.classIndex();
		double[] response=data.attributeToDoubleArray(responsePos);
		double v;
		for(int i=0;i<response.length;i++)
		{
			inst=data.instance(i);

			
			if(response[i]<0)
			{
				if(lambda!=2)
					v=-(Math.pow((1-response[i]),2-lambda)-1)/(2-lambda);
				else
					v=-Math.log(1-response[i]);
			}
			else
			{
				if(lambda==0)
					v=Math.log(1+response[i]);
				else
					v=(Math.pow(response[i]+1,lambda)-1)/lambda;
			}
			inst.setValue(responsePos,v);
		}
		
		return data;
	}
	
	static public Instances transformResponse(Instances data, double lambda)
	{
		transformResponse(data,lambda,data.attributeToDoubleArray(data.classIndex()));
		return data;
	}
	public static double[] invert(double lambda, double[] response)
	{
		double[] data=new double[response.length];
		for(int i=0;i<response.length;i++)
		{
			if(response[i]<0)
			{
				if(lambda!=2)
				{
//					data[i]=-(Math.pow((1-response[i]),2-lambda)-1)/(2-lambda);
	//Need to check whether 1.0/(2.0-lambda) is negative I think, as it doesnt work wth power?				
					data[i]=1-Math.pow((1-(2-lambda)*response[i]),1.0/(2.0-lambda));
				}
				else
//					data[i]=-Math.log(1-response[i]);
					data[i]=1-Math.exp(-response[i]);
			}
			else
			{
				if(lambda==0)
//					data[i]=Math.log(1+response[i]);
					data[i]=Math.exp(response[i])-1;
				else
//					data[i]=(Math.pow(response[i]+1,lambda)-1)/lambda;
				{//HACK
					if(lambda*response[i]+1>0.000001)
						data[i]=Math.pow((lambda*response[i]+1),1.0/lambda)-1;
					else if(i==0)
						data[i]=Math.pow((0.000001),1.0/lambda)-1;
					else
						data[i]=1.05*data[i-1];
				}
			}
			if(data[i]==Double.NaN)
			{
				System.out.println("NAN in invert: Response = "+response[i]+" lambda = "+lambda);
				System.exit(0);	
			}
		}
		return data;
	}
	public static double[] transform(double lambda, double[] response)
	{
		double[] data=new double[response.length];
		for(int i=0;i<response.length;i++)
		{
			if(response[i]<0)
			{
				if(lambda!=2)
					data[i]=-(Math.pow((1-response[i]),2-lambda)-1)/(2-lambda);
				else
					data[i]=-Math.log(1-response[i]);
			}
			else
			{
				if(lambda==0)
					data[i]=Math.log(1+response[i]);
				else
					data[i]=(Math.pow(response[i]+1,lambda)-1)/lambda;
			}
			if(data[i]==Double.NaN)
			{
				System.out.println("NAN in transform: Response = "+response[i]+" lambda = "+lambda);
				System.exit(0);	
			}
		}
		return data;
	}
	static public Instances transformInstances(Instances data, double lambda)
	{
		transformResponse(data,lambda,data.attributeToDoubleArray(data.classIndex()));
		return data;
	}

	
	static public void transformResponse(Instances data, double lambda, double[] response)
	{
		Instance inst;
		int responsePos=data.classIndex();
		double[] newData=transform(lambda,response);
		for(int i=0;i<response.length;i++)
		{
			inst=data.instance(i);
			inst.setValue(responsePos,newData[i]);
		}
	}
	static public double findBestTransform(double[][] data, double[] res)
	{
		double[] response;

//		double[] predictions=new double[res.length];	
		double v;
		Instance inst;
		LinearModel lm;
		double bestLambda=MIN,minError=Double.MAX_VALUE,error;
		double correlation;
		for(double lambda=MIN;lambda<=MAX;lambda+=INTERVAL)
		{
//Transform response				
			response=transform(lambda,res);
			lm=new LinearModel(data,response);
			lm.fitModel();
			double e =lm.findStats();
//Initially, just going to use standardised SSR

/*Use the K-S stat for this		
			double ks=ResidualTests.kolmogorovSmirnoff(lm.stdResidual);
			correlation=ResidualTests.testHeteroscadisity(lm.y,lm.predicted);
*/			error=lm.findInverseStats(lambda,res);
//			error=correlation;
			if(error<minError)
			{
				bestLambda=lambda;
				minError=error;
			}
//			System.out.println(" Lambda ="+lambda+" untransformed error = "+e+" Transformed error = "+error+" Correlation = "+correlation+" KS = "+ks);
		}
//		power[pos]=bestLambda;
		return bestLambda;
	}

	static public double findBestTransform(Instances data, int pos, double[] power)
	{
		int responsePos=data.classIndex();
		double[] temp=data.attributeToDoubleArray(responsePos);
		double[] response=new double[temp.length];
                System.arraycopy(temp, 0, response, 0, temp.length);
		double[] predictions=new double[response.length];	
		double v;
		Instance inst;
		LinearModel lm;
		double bestLambda=MIN,minError=Double.MAX_VALUE,error;
		double correlation;
		for(double lambda=MIN;lambda<=MAX;lambda+=INTERVAL)
		{
//Transform response				
			transformResponse(data,lambda,response);
			lm=new LinearModel(data);
			lm.fitModel();
			lm.formTrainPredictions();
			lm.findTrainStatistics();
			
//Use the K-S stat for this		
			error=ResidualTests.kolmogorovSmirnoff(lm.stdResidual);
			correlation=ResidualTests.testHeteroscadisity(lm.y,lm.predicted);
			if(error<minError)
			{
				bestLambda=lambda;
				minError=error;
			}
//			System.out.println(" Lambda ="+lambda+" KS Stat = "+error+" Correlation = "+correlation);
		}
		power[pos]=bestLambda;
		return minError;
	}
    @Override 
	public Instances transform(Instances data)
	{
            System.out.println(" Doesnt do anything! ");
		int responsePos=data.numAttributes()-1;
		double[] response=data.attributeToDoubleArray(responsePos);
		double[] preds=new double[response.length];	
		double v;
		Instance inst;
		return data;
	}
	public static void main(String[] args)
	{
		int size=100;
		double[] d=new double[size];
		for(int i=0;i<size;i++)
			d[i]=-5+0.1*i;
		OutFile f= new OutFile("TestYeoJohnson.csv");
		f.writeLine("Data,Lambda-3,Lambda-1,Lambda-0.5,Lambda0,Lambda0.5,Lambda1,Lambda2,Lambda3, InvLambda-3,InvLambda-1,InvLambda-0.5,InvLambda0,InvLambda0.5,InvLambda1,InvLambda2,InvLambda3");
		double[][] d2=new double[8][];
		double[][] inv=new double[8][];
		d2[0]=transform(-3.0,d);
		d2[1]=transform(-1.0,d);
		d2[2]=transform(-0.5,d);
		d2[3]=transform(0.0,d);
		d2[4]=transform(0.5,d);
		d2[5]=transform(1.0,d);
		d2[6]=transform(2.0,d);
		d2[7]=transform(3.0,d);
		inv[0]=invert(-3.0,d2[0]);
		inv[1]=invert(-1.0,d2[1]);
		inv[2]=invert(-0.5,d2[2]);
		inv[3]=invert(0.0,d2[3]);
		inv[4]=invert(0.5,d2[4]);
		inv[5]=invert(1.0,d2[5]);
		inv[6]=invert(2.0,d2[6]);
		inv[7]=invert(3.0,d2[7]);

		
		for(int i=0;i<size;i++)
		{
			f.writeString(d[i]+",");
			for(int j=0;j<8;j++)
				f.writeString(d2[j][i]+",");
			for(int j=0;j<8;j++)
				f.writeString(inv[j][i]+",");
			f.writeString("\n");
		}
	}
}