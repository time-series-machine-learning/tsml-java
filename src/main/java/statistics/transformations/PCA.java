/*
 * Created on Jan 30, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package transformations;

import java.io.FileReader;
import fileIO.*;
import weka.core.*;
import weka.attributeSelection.PrincipalComponents;

public class PCA extends Transformations {
	
	double varianceCovered=1;
        public void setVariance(double v){
            varianceCovered=v;
        }
	PrincipalComponents pca=new PrincipalComponents();
	
	public Instances transform(Instances data){
		Instances newData=data;
		try{
			pca.setVarianceCovered(varianceCovered);
			pca.buildEvaluator(data);
			newData=pca.transformedData(data);
		}catch(Exception e)
		{
			System.out.println(" Error = "+e);
			System.exit(0);
		}
		
		//Build the transformation
		
	//Add the response back in	
		return newData;
	}
//Generally dont need this	
	public Instances invert(Instances data){
		return data;
	}

	public Instances staticTransform(Instances data){
		Instance inst=null;
		Instances newData=null;
		
		try{
			newData=pca.transformedHeader();
			for(int i=0;i<data.numInstances();i++)
			{
				inst=pca.convertInstance(data.instance(i));
				newData.add(inst);
			}
		}catch(Exception e)
		{
			System.out.println(" Error in convert "+e);
			System.out.println(" instance ="+inst);
			System.exit(0);
		}
		return newData;
	
	}
//PCA Leaves the response the same	
	public double[] invertPredictedResponse(double[] d){
		return d;
	}
	public static void main(String[] args){
		PCA p= new PCA();
		Instances data;
		FileReader r;
		try{	
			r= new FileReader("C:/Research/Data/Gavin Competition/Weka Files/SO2Combined.arff");
//			r= new FileReader("C:/Research/Data/Gavin Competition/Weka Files/Temp Train.arff");
			data = new Instances(r); 
			data.setClassIndex(data.numAttributes()-1);
			p.varianceCovered=0.95;
			data=p.transform(data);
			System.out.println(" New attribute size = "+data.numAttributes());
			OutFile of= new OutFile("C:/Research/Data/Gavin Competition/Weka Files/SO2CombinedTransformed.arff");
			of.writeLine(data.toString());
			r= new FileReader("C:/Research/Data/Gavin Competition/Weka Files/PrecipCombined.arff");
//			r= new FileReader("C:/Research/Data/Gavin Competition/Weka Files/Temp Train.arff");
			data = new Instances(r); 
			data.setClassIndex(data.numAttributes()-1);
			p.varianceCovered=0.95;
			data=p.transform(data);
			System.out.println(" New attribute size = "+data.numAttributes());
			of= new OutFile("C:/Research/Data/Gavin Competition/Weka Files/PrecipCombinedTransformed.csv");
			of.writeLine(data.toString());
		}catch(Exception e)
		{
			System.out.println(" Error in PCA "+e);
		}
	}
	
}
