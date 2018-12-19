/*

Basic linear regression, including standardisation for residuals which isnt included
in the Jama ridge regression


 * */
package transformations;

import fileIO.OutFile;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.matrix.Matrix;

public class LinearModel {
	double variance, standardisedError,SSE,SST,SSR, yBar;
	Matrix Xt,X,XtXinv;
	Matrix Y;
	Matrix B;
	double[] paras,y,H_Diagonal,predicted,residual,stdResidual;
	int n,m;
	
//H is going to be big! X(XtX)-1Xt is nxn, so need to just generate diagonal terms
//	X(XtX)-1 is nxm, so can just work the diagonals with Xt  	 
	
	public Matrix HatDiagonal;
	
//ASSUMES FIRST ROW IS ALL ONES IF CONSTANT TERM TO BE INCLUDED	
//ATTRIBUTE FIRST Dirty hack	
	public LinearModel(double[][] data,double[] response)	
	{
		m=data.length;
		n=data[0].length;
		y = response;
//This way round for consistency with other constructor
		Xt=new Matrix(data);
//		System.out.println("Xt = \n"+Xt);
		X = Xt.transpose();
//		System.out.println("X = \n"+X);
		Y=new Matrix(y,y.length);
	}
	public LinearModel(Instances data)
	{
//Form X and Y from Instances		
		n=data.numInstances();
		m=data.numAttributes();	//includes the constant term
		y = data.attributeToDoubleArray(data.classIndex());
		Y=new Matrix(y,y.length);
		double[][] xt = new double[m][n];
		for(int i=0;i<n;i++)
			xt[0][i]=1;
		for(int i=1;i<m;i++)
			xt[i]=data.attributeToDoubleArray(i-1);
		Xt=new Matrix(xt);
		X=Xt.transpose();
	}
	
	public double[] getY(){return y;}
	public double[] getPredicted(){return predicted;}
	public double[] getResiduals(){return stdResidual;}
	public double getSSR(){return SSR;}
	
	public void fitModel()
	{
//B = (XtX)-1XtY		
		XtXinv=Xt.times(X);
		XtXinv=XtXinv.inverse();
		Matrix temp= XtXinv.times(Xt),t2,t3;
//B should be m x 1		
		B=temp.times(Y);
		paras=B.getColumnPackedCopy();
		H_Diagonal=new double[n];
//		(XtX)-1Xt is mxn, so can just work the diagonals with Xt  	 
		double sum=0;
		for(int i=0;i<n;i++)
		{
			t2=X.getMatrix(i,i,0,m-1);
			t3=t2.transpose();
//			System.out.println("Row mult t2 rows ="+t2.getRowDimension()+" columns = "+t2.getColumnDimension());
			t3=XtXinv.times(t3);
			t3=t2.times(t3);
			H_Diagonal[i]=t3.get(0,0);
			sum+=H_Diagonal[i];
		}
		
	}

	public double findInverseStats(double l, double[] untransformed)
	{
		formTrainPredictions();
		predicted=YeoJohnson.invert(l,predicted);
		y=untransformed;
		findTrainStatistics();
		return variance;
	}
	public double findStats()
	{
		formTrainPredictions();
		findTrainStatistics();
		return variance;
	}
	public  double[] formTrainPredictions()
	{
		predicted=new double[n];
		for(int i=0;i<n;i++)
		{
			//Find predicted
			predicted[i]=paras[0];
			for(int j=1;j<paras.length;j++)
				predicted[i]+=paras[j]*X.get(i,j);
		}
		return predicted;
	}
	public  void findTrainStatistics()
	{
		SSE=0;
		stdResidual=new double[n];
		residual=new double[n];
		yBar=0;
		for(int i=0;i<n;i++)
		{
			residual[i]=(y[i]-predicted[i]);
			SSE+=residual[i]*residual[i];
			yBar+=y[i];
		}
		yBar/=n;
		variance=SSE/(n-paras.length);
		SST=0;
		for(int i=0;i<n;i++)
			SST+=(y[i]-yBar)*(y[i]-yBar);
		SSR=SST-SSE;
		double s= Math.sqrt(variance);
		standardisedError=0;
		for(int i=0;i<n;i++)
		{
			stdResidual[i]=residual[i]/(s*(Math.sqrt(1-H_Diagonal[i])));
			standardisedError+=stdResidual[i]*stdResidual[i];
		}
		standardisedError/=(n-paras.length);
	}
	public  double[] formTestPredictions(Instances testData)
	{
//Form X matrix from testData
		int rows=testData.numInstances();
		int cols=testData.numAttributes();	//includes the constant term
		predicted=new double[rows];
		if(cols!=m)
		{
			System.out.println("Error: Mismatch in attribute lengths in form test Train ="+m+" Test ="+cols);
			System.exit(0);
		}
		double[][] xt = new double[cols][rows];
		for(int i=0;i<rows;i++)
			xt[0][i]=1;
		for(int i=1;i<cols;i++)
			xt[i]=testData.attributeToDoubleArray(i-1);
		Matrix testX=new Matrix(xt);
		testX=testX.transpose();
		
		for(int i=0;i<rows;i++)
		{
			//Find predicted
			predicted[i]=paras[0];
			for(int j=1;j<paras.length;j++)
				predicted[i]+=paras[j]*testX.get(i,j);
		}
		return predicted;
	
	}
	public String toString()
	{
		String str="Paras : ";
		for(int i=0;i<paras.length;i++)
			str+=paras[i]+" ";
		return str;
	}
	public static void main(String[] args) {
		Instances data=null;
		try{
			FileReader r = new FileReader("C:/Research/Code/Archive Generator/src/weka/addOns/RegressionTest2.arff");
			data = new Instances(r);
			data.setClassIndex(data.numAttributes()-1);
		}catch(Exception e)
		{
			System.out.println("Error loading file "+e);
		}
		LinearModel lm = new LinearModel(data);
		lm.fitModel();
		lm.formTrainPredictions();
		lm.findTrainStatistics();
		OutFile f = new OutFile("C:/Research/Code/Archive Generator/src/weka/addOns/TestResults.csv");
		f.writeLine("Parameters");
		for(int i=0;i<lm.paras.length;i++)
			f.writeString(lm.paras[i]+",");
		f.writeLine("Variance = "+lm.variance);
		f.writeLine("\nHatDiagonal, Actual, Predicted, StdResidual");
		for(int i=0;i<lm.n;i++)
			f.writeLine(lm.H_Diagonal[i]+","+lm.y[i]+","+lm.predicted[i]+","+lm.stdResidual[i]);
		
		
		
	}

    public double getSSE() {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
