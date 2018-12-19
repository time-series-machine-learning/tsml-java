package timeseriesweka.filters;

import weka.core.*;
import weka.core.matrix.*;
import weka.filters.*;
import weka.core.matrix.Matrix;

/* 6/3/13: THIS IS NOT WORKING CORRECTLY, DO NOT USE This filter transforms the data according to Jans idea. Not sure 
 * 
 * <xy>-<x><y>
   -------------- = r = correlation coefficient
    std(x)*std(y)


 */
public class Correlation extends SimpleBatchFilter {
	Matrix covariance;
	Matrix eigenvectors;
	Matrix X;
	int numAtts=0;
	
	private void findCovariance(Instances inst){

		setOutputFormat(inst);
//! Extract data set as a matrix X
		numAtts=inst.numAttributes();
		int c= inst.classIndex();
		if(c>0)
			numAtts--;
//If a classification problem it is assumed the class attribute is the last one in the instances
		double[][] d =new double[inst.numInstances()][numAtts];
		for(int i=0;i<inst.numInstances();i++)
		{
			Instance x=inst.instance(i);
			for(int j=0;j<numAtts;j++)
				d[i][j]=x.value(j);
		}
		//Find means and subtract from X
		double mean;
		for(int j=0;j<numAtts;j++)
		{
			mean=0;
			for(int i=0;i<inst.numInstances();i++)
				mean+=d[i][j];
			mean/=inst.numInstances();
			for(int i=0;i<inst.numInstances();i++)
				d[i][j]-=mean;
			
		}
		X=new Matrix(d);
		//Work out X_T X		
		covariance=X.transpose().times(X);	
		covariance.timesEquals(1.0/((double)inst.numInstances()-1));
	}
	
	public void findEigenVectors(){
	//Pull out the Eigenvectors
		EigenvalueDecomposition ev = new EigenvalueDecomposition(covariance);
		eigenvectors=ev.getV();
	}
	
	public Instances process(Instances inst) throws Exception {
            Instances result=determineOutputFormat(inst);
            findCovariance(inst);
            findEigenVectors();
            Matrix y=eigenvectors.times(X);
                //Stuff into new set of instances
            for(int i=0;i<inst.numInstances();i++) {
            //Create a deep copy, think this is necessary to maintain meta data?
                    Instance in=new DenseInstance(inst.instance(i)); 
            //Reset to the new vals
                    for(int j=0;j<numAtts;j++)
                            in.setValue(j, y.get(i, j));
                    result.add(in);
            }
            return result;			  
        }


	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
            
            if(true)
                throw new Exception("Added: 6/3/13. FILTER CORRELTATION IS NOT CORRECTLY IMPLEMENTED, DO NOT USE");            
              int length=inputFormat.numAttributes();	
                if(inputFormat.classIndex()>=0)
                    length--;
                //Check all attributes are real valued, otherwise throw exception
                for(int i=0;i<inputFormat.numAttributes();i++)
                        if(inputFormat.classIndex()!=i)
                                if(!inputFormat.attribute(i).isNumeric())
                                        throw new Exception("Non numeric attribute not allowed in Correlation Filter");
                //Set up instances size and format. 
                FastVector atts=new FastVector();
                String name;
                for(int i=0;i<length;i++){
                        name = "Correlation_"+i;
                        atts.addElement(new Attribute(name));
                }
                if(inputFormat.classIndex()>=0){	//Classification set, set class 
                        //Get the class values as a fast vector			
                        Attribute target =inputFormat.attribute(inputFormat.classIndex());

                        FastVector vals=new FastVector(target.numValues());
                        for(int i=0;i<target.numValues();i++)
                                vals.addElement(target.value(i));
                        atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
                }	
                Instances result = new Instances("Correlation"+inputFormat.relationName(),atts,inputFormat.numInstances());
                if(inputFormat.classIndex()>=0){
                        result.setClassIndex(result.numAttributes()-1);
                }

                return result;
        
        }
	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public static void main(String[] args){
		
		double[][] d = {{1,2,1},{6,-1,0},{-1,-2,-1}};
		
		Matrix m=new Matrix(d);
		EigenvalueDecomposition ev = new EigenvalueDecomposition(m);
		Matrix eigenvectors=ev.getV();
		double[] evals=ev.getRealEigenvalues();
		for(double x:evals)
			System.out.println(x);
		System.out.println(eigenvectors);
		Matrix y=ev.getD();
		System.out.println(y);
		
		System.exit(0);
		
//		Instances data =SRNACluster.loadData(SRNACluster.rootPath+"sRNA10_small");
//		Filter ct=new Correlation();
//		RankOrderTransform.testFilter(data,ct); 
	}

}
