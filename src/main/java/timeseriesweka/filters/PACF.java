/*
     * copyright: Anthony Bagnall
 * */
package timeseriesweka.filters;

import weka.filters.*;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class PACF extends SimpleBatchFilter {
//Max number of AR terms to consider. 	
    public static int globalMaxLag=25;
    private double[] autos;
    private double[][] partials;

//Defaults to 1/4 length of series
    public int maxLag=globalMaxLag;

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
                    name = "PACF_"+i;
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
            Instances result = new Instances("PACF"+inputFormat.relationName(),atts,inputFormat.numInstances());
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
                    partials=formPartials(autos);
            //5. Find parameters
                    double[] pi= new double[maxLag];
                    for(int k=0;k<maxLag;k++){  //Set NANs to zero
                        if(Double.isNaN(partials[k][k]) || Double.isInfinite(partials[k][k])){
                            pi[k]=0;
                        }
                        else
                            pi[k]=partials[k][k];
                    }
            //6. Stuff back into new Instances.
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
    public static double[][] formPartials(double[] r){
    //Using the Durban-Leverson
            int p=r.length;
            double[][] phi = new double[p][p];
            double numerator,denominator;
            phi[0][0]=r[0];

            for(int k=1;k<p;k++){
//Find diagonal k,k
//Naive implementation, should be able to do with running sums
                    numerator=r[k];
                    for(int i=0;i<k;i++)
                            numerator-=phi[i][k-1]*r[k-1-i];
                    denominator=1;
                    for(int i=0;i<k;i++)
                            denominator-=phi[k-1-i][k-1]*r[k-1-i];
                    phi[k][k]=numerator/denominator;
            //Find terms 1,k to k-1,k
                    for(int i=0;i<k;i++)
                            phi[i][k]=phi[i][k-1]-phi[k][k]*phi[k-1-i][k-1];
            }
            return phi;
    }


    public double[][] getPartials(){return partials;}

    @Override
    public String globalInfo() {
            return null;
    }

    @Override
    protected boolean hasImmediateOutputFormat() {
            return false;
    }
    public String getRevision() {
            return null;
    }

    public static void main(String[] args){





    }
	
	
}
