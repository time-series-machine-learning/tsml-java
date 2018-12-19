package statistics.tests;

import statistics.distributions.FisherDistribution;
import statistics.distributions.NormalDistribution;
import fileIO.OutFile;
import java.io.FileReader;
import java.util.Arrays;
import transformations.ArrayPair;
import transformations.LinearModel;
import transformations.MatrixSort;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class to test residuals. Dont know where classes MatrixSort and LinearModel are!
 */
public class ResidualTests {


//Returns the test stat	
	public static double goldfeldQuandt(double[][]X, double[] Y, int pos)
	{
//Copy data and sort by selected attribute
		MatrixSort[] ms = new MatrixSort[Y.length];
		for(int i=0;i<Y.length;i++)
		{
			double[] x = new double[X.length];
			for(int j=0;j<x.length;j++)
				x[j]=X[j][i];
			ms[i]=new MatrixSort(x,Y[i],pos);
		}
		Arrays.sort(ms);
//Split into three sets size n1, n2=n/4 and n3
		int p=X.length-1;
		int n=Y.length;
		int n2=n/3;
		int n1=(n-n2)/2;
		int n3=n-n1-n2;
		System.out.println("n1 = "+n1+" n2 = "+n2+" n3 = "+n3);
		double[][]newX = new double[X.length][n1];
		double[] newY= new double[n1];
		for(int i=0;i<n1;i++)
		{
			newY[i]=Y[i];
			for(int j=0;j<X.length;j++)
				newX[j][i]=ms[i].x[j];
		}
//		Fit regression to first set, find SSE

		LinearModel lm =new LinearModel(newX,newY);
		lm.fitModel();
		lm.findStats();
		double s1=lm.getSSE();
		
//		Fit regression to second set, find SSE
		newX = new double[X.length][n3];
		newY= new double[n3];
		for(int i=n1+n2;i<n;i++)
		{
			newY[i-n1-n2]=Y[i];
			for(int j=0;j<X.length;j++)
				newX[j][i-n1-n2]=ms[i].x[j];
		}
		lm =new LinearModel(newX,newY);
		lm.fitModel();
		lm.findStats();
		double s3=lm.getSSE();
//Find q = (n1-p-1)SSE1/(n3-p-1)SSE3		
		
		double q;
		q=((n3-p-1)*s1)/((n1-p-1)*s3);
			
		System.out.println(" s1 = "+s1+"  s3 = "+s3+"  q = "+q+" s3/s1"+s3/s1);
		FisherDistribution f;
		if(s1>s3)
			f = new FisherDistribution(n3-p-1, n1-p-1);
		else
			f= new FisherDistribution(n3-p-1, n1-p-1);
		double prob = f.getCDF(q);
		double prob2=f.getDensity(q);
		System.out.println(" prob = "+prob+" density = "+prob2);
		
	//This follows an F(n1-p-1,k3-p-1) distribution under the null of homoscedastic	
		return q;
	}
	
	public static double runsTest(double[] pred, double[] residuals)
	{
		double runsCount=1;
		double p=0;
		boolean positive=false, currentPositive;
		ArrayPair[] ap = new ArrayPair[pred.length];
		for(int i=0;i<pred.length;i++)
		{
			ap[i]=new ArrayPair();
			ap[i].predicted=pred[i];
			ap[i].residual=residuals[i];
		}
		Arrays.sort(ap);
		
		if(ap[0].residual>0)
		{
			positive=true;
			p=1;
		}
		for(int i=1;i<ap.length;i++)
		{
			if(ap[i].residual>0)
			{
				currentPositive=true;
				p++;
			}
			else
				currentPositive=false;
			if(currentPositive!=positive)
				runsCount++;
			positive=currentPositive;
		}
		double n=ap.length;
//		System.out.println("Runs count = "+runsCount+" number of ones = "+p);
		//Calculate probs via normal
		double m=(2.0*p*(n-p))/(n-1);

//Something wrong with v!?!
		double v=(2*p*(n-p)*(2*p*(n-p)-n))/(n*n*(n-1));
		System.out.println("m = "+m+" v = "+v);
//Better to use the weka normal distribution		
		double res=(runsCount-m)/Math.sqrt(v);
		return res;
		
	}
	public static double kolmogorovSmirnoff(double[] residuals)
	{
		return kolmogorovSmirnoff(residuals,1);
	}
	public static double kolmogorovSmirnoff(double[] residuals, double var)
	{
//Normality test for residuals: Kolmogorov Smirnoff		
		int n=residuals.length;
		double[] expected=new double[n];
		double[] observed=new double[n+1];
		double[] residCopy= new double[residuals.length];
        System.arraycopy(residuals, 0, residCopy, 0, residuals.length);
		observed[n]=1;
		NormalDistribution norm = new NormalDistribution(0,var);

		Arrays.sort(residCopy);
//Find out the Expected normal values for the stepped probabilities
		//Set probs
		for(int i=0;i<n;i++)
			expected[i]=(i+1)/(double)n;
		//Find inverses
		for(int i=0;i<n;i++)
			observed[i]=norm.getCDF(residCopy[i]);
//Find max deviation
		double max=0;
		for(int i=0;i<n;i++)
		{
			if(Math.abs(expected[i]-observed[i+1])>max)
				max=Math.abs(expected[i]-observed[i+1]);
		}
		return max;
	}

	public static double anscombeProcedure(double[] actual, double[] predicted)
	{
		return testHeteroscadisity(actual,predicted);
	}
	public static double testHeteroscadisity(double[] actual, double[] predicted)
	{
//Measure correlation between actual values and absolute residual values
		double[] absRes = new double[predicted.length];
		double meanPred=0;
		double meanAbs=0;
		for(int i=0;i<predicted.length;i++)
		{
			absRes[i]=Math.abs(actual[i]-predicted[i]);
			meanAbs+=absRes[i];
			meanPred+=predicted[i];
		}
		meanAbs/=predicted.length;
		meanPred/=predicted.length;
		//Measure correlation between absRes and predicted, quite slowly!
		double corr=0,x=0,y=0;
		for(int i=0;i<predicted.length;i++)
		{
			corr+=(absRes[i]-meanAbs)*(predicted[i]-meanPred);
			x+=(absRes[i]-meanAbs)*(absRes[i]-meanAbs);
			y+=(predicted[i]-meanPred)*(predicted[i]-meanPred);
		}
		corr=corr/Math.sqrt(x*y);
		System.out.println(" Correlation = "+corr);
//Not adjusted for the number of regressors!		
//		double t=corr*(Math.sqrt(predicted.length-2))/Math.sqrt(1-corr*corr);
		return corr;
	}

	public static void main(String[] args) {

		int s=100;
		double[] data= new double[s];
		NormalDistribution n = new NormalDistribution(0,3);
		for(int i=0;i<s;i++)
			data[i]=n.simulate();
		double x = kolmogorovSmirnoff(data,1);
		System.out.println(" KS stat = "+x);

	}
	public static void testHetero()
	{
		Instances data;
		FileReader r;
		Instance inst;
		double[] actual,predictions;
		LinearRegression lg = new LinearRegression();
		try{	
			r= new FileReader("C:/Research/Data/Gavin Competition/Weka Files/Temp Train.arff");
			data = new Instances(r); 
			data.setClassIndex(data.numAttributes()-1);
			lg.buildClassifier(data);
			predictions=new double[data.numInstances()];
			actual=data.attributeToDoubleArray(data.numAttributes()-1);
			for(int i=0;i<predictions.length;i++)
			{
				inst=data.instance(i);
				predictions[i]=lg.classifyInstance(inst);
			}
			
			OutFile of= new OutFile("C:/Research/Data/Gavin Competition/Weka Files/CorrelationTest.csv");
			System.out.println(" t stat for homogeneity ="+testHeteroscadisity(actual,predictions));
			for(int i=0;i<predictions.length;i++)
				of.writeLine(actual[i]+","+predictions[i]);
		}catch(Exception e)
		{
			System.out.println(" Error in REsidual Test "+e);
		}
	}
}
