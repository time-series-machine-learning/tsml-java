/*
 *      * copyright: Anthony Bagnall

 */
package timeseriesweka.filters;

import weka.filters.*;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class ARMA extends SimpleBatchFilter {
//Max number of AR terms to consider. 	
	double[] ar;
	public static int globalMaxLag=25;
//Defaults to 1/4 length of series
	public int maxLag=globalMaxLag;
	public boolean useAIC=true;
	
	public void setUseAIC(boolean b){useAIC=b;}
	
	public void setMaxLag(int a){maxLag=a;}
	
	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		//Check all attributes are real valued, otherwise throw exception
		for(int i=0;i<inputFormat.numAttributes();i++)
			if(inputFormat.classIndex()!=i)
				if(!inputFormat.attribute(i).isNumeric())
					throw new Exception("Non numeric attribute not allowed in ACF");

		if(inputFormat.classIndex()>=0)	//Classification set, dont transform the target class!
			maxLag=(inputFormat.numAttributes()-1>maxLag)?maxLag:inputFormat.numAttributes()-1;
		else
			maxLag=(inputFormat.numAttributes()>maxLag)?maxLag:inputFormat.numAttributes();
		//Set up instances size and format. 
		FastVector atts=new FastVector();
		String name;
		for(int i=0;i<maxLag;i++){
			name = "ARMA_"+i;
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
		Instances result = new Instances("ARMA"+inputFormat.relationName(),atts,inputFormat.numInstances());
		if(inputFormat.classIndex()>=0)
			result.setClassIndex(result.numAttributes()-1);
		return result;	}

	@Override
	public Instances process(Instances inst) throws Exception {
		Instances output=determineOutputFormat(inst);
		
		//For each data, first extract the relevan
		int seriesLength=inst.numAttributes();
		int acfLength=output.numAttributes();
		if(inst.classIndex()>=0){
			seriesLength--;
			acfLength--;
		}
		double[] d;
		double[] autos;
		double[][] partials;
		for(int i=0;i<inst.numInstances();i++){
		//1. Get series
			d=inst.instance(i).toDoubleArray();
			//Need to remove the class
			int c=inst.classIndex();
			if(c>=0){
				double[] temp=new double[d.length-1];
				int count=0;
//Arraycopy more efficient, dont really trust it
				for(int k=0;k<d.length;k++){
					if(k!=c){
						temp[count]=d[k];
						count++;
					}
				}
				d=temp;
			}
		//2. Fit Autocorrelations
			autos=ACF.fitAutoCorrelations(d,maxLag);
		//3. Form Partials	
			partials=PACF.formPartials(autos);
		//4. Find bet AIC. Could also use BIC?
			int best=maxLag;
			if(useAIC)
				best=findBestAIC(autos,partials,maxLag,d);
		//5. Find parameters
			double[] pi= new double[maxLag];
			for(int k=0;k<best;k++)
				pi[k]=partials[k][best-1];
		//6. Stuff back into new Instances. NEED TO CLONE IT
			Instance in= new DenseInstance(output.numAttributes());
			//Set class value.
			int cls=output.classIndex();
			if(cls>=0)
				in.setValue(cls, inst.instance(i).classValue());
			int count=0;
			//Allows for a class index not at the end, or should do so.
			for(int k=0;k<pi.length;k++){
				if(k!=cls){
					in.setValue(count, pi[k]);
					count++;
				}
			}
			output.add(in);
			
		}
		return output;
	}
	
	public static double[] fitAR(double[] d){
		//2. Fit Autocorrelations
		double[] autos=ACF.fitAutoCorrelations(d,globalMaxLag);
	//3. Form Partials	
		double[][] partials=PACF.formPartials(autos);
	//4. Find bet AIC. Could also use BIC?	
		int best=findBestAIC(autos,partials,globalMaxLag,d);
	//5. Find parameters
		double[] pi= new double[globalMaxLag];
		for(int k=0;k<best;k++)
			pi[k]=partials[k][best-1];		
		return pi;
	}
	
	
	public static int findBestAIC(double[] autoCorrelations, double[][] partialCorrelations, int maxLag, double[] d)
	{
// need the variance of the series
		double sigma2;
		int n=d.length;
		double var=0, mean=0;
		for(int i=0;i<d.length;i++)
			mean+=d[i];
		for(int i=0;i<d.length;i++)
			var+=(d[i]-mean)*(d[i]-mean);
		var/=(d.length-1);
		double AIC=Double.MAX_VALUE;
		double bestAIC=Double.MAX_VALUE;
		int bestPos=0;
		int i=0;
		boolean found=false;
		while(i<maxLag && !found){
			sigma2=1;
			for(int j=0;j<=i;j++){
				sigma2-=autoCorrelations[j]*partialCorrelations[j][i];
//				System.out.println("\tStep ="+j+" incremental sigma ="+sigma2);
			}
			sigma2*=var;
			AIC=Math.log(sigma2);
			i++;
			AIC+=((double)2*(i+1))/n;
//			System.out.println("LAG ="+i+"  final sigma = "+sigma2+" log(sigma)="+Math.log(sigma2)+" AIC = "+AIC);
           if(AIC==Double.NaN)
               AIC=Double.MAX_VALUE;
			if(AIC<bestAIC){
				bestAIC=AIC;
				bestPos=i;
			}
			else
				found=true;
		}
		return bestPos;
	}

	
	
	@Override
	public String globalInfo() {
		return null;
	}


	public String getRevision() {
		return null;
	}

/* This function verifies the output is the sam as from R
 The R code to perform the ACF, ARMA and PACF comparison is in 
 
 
 */
        
  /*      
        public static void testTransform(String path){
            Instances data=ClassifierTools.loadData(path+"ACFTest");
            ACF acf=new ACF();
            acf.setNormalized(false);
            PACF pacf=new PACF();
            ARMA arma=new ARMA();
            int lag=10;
            acf.setMaxLag(lag);
            pacf.setMaxLag(lag);
            arma.setMaxLag(lag);
            arma.setUseAIC(false);
            try{
            Instances acfD=acf.process(data);
            Instances pacfD=pacf.process(data);
            Instances armaD=arma.process(data);
//Save first case to file
            OutFile of=new OutFile(path+"ACFTest_JavaOutput.csv");
            of.writeLine(",acf1,pacf1,arma");
            for(int i=0;i<acfD.numAttributes()-1;i++)
                of.writeLine("ar"+(i+1)+","+acfD.instance(0).value(i)+","+pacfD.instance(0).value(i)+","+armaD.instance(0).value(i));
            double[][] partials=pacf.getPartials();
            of.writeLine("\n\n");
            for(int i=0;i<partials.length;i++){
                of.writeString("\n");
                for(int j=0;j<partials[i].length;j++)
                    of.writeString(partials[i][j]+",");
                    
            }
                
            }
            catch(Exception e){
                System.out.println("Exception caught, exit "+e);
                e.printStackTrace();
                System.exit(0);
            }
        }
	public static void main(String[] args){
            
         testTransform("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\");
         System.exit(0);
//Debug code to test. 
		ARMA ar = new ARMA();
		ar.setUseAIC(false);
		
		//Generate a model
		double[][] paras={{0.5},{0.7}};
		int n=100;
		int cases=1;
//		double[][] paras={{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004},
//		{1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019}	};	

		//Generate a series
		
		//Fit and compare paramaters without AIC
		
		//Fit using AIC and compare again
		
		
	}
*/	
	
}
