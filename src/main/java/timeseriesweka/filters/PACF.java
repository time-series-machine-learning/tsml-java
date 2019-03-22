/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */ 
package timeseriesweka.filters;

import java.util.ArrayList;
import weka.filters.*;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
/** 
  <!-- globalinfo-start -->
* Implementation of partial autocorrelation function as a Weka SimpleBatchFilter
* Series to series transform independent of class value
*  <!-- globalinfo-end -->
<!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -L
 *  set the max lag.</pre>
 <!-- options-end -->
 *
 * 
 * author: Anthony Bagnall circa 2008.
 * Reviewed and tidied up 2019
 * This should not really be a batch filter, as it is series to series, but
 * it makes the use case simpler. 
 */

public class PACF extends SimpleBatchFilter {

    public static final int DEFAULT_MAXLAG=100;
    private double[] autos;
    private double[][] partials;

//Defaults to 1/4 length of series
    public int maxLag=DEFAULT_MAXLAG;

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
        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int i=0;i<maxLag;i++){
            name = "PACF_"+i;
            atts.add(new Attribute(name));
        }
        if(inputFormat.classIndex()>=0){	//Classification set, set class 
//Get the class values 		
            Attribute target =inputFormat.attribute(inputFormat.classIndex());
            ArrayList<String> vals=new ArrayList<>(target.numValues());
            for(int i=0;i<target.numValues();i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
        }	
        Instances result = new Instances("PACF"+inputFormat.relationName(),atts,inputFormat.numInstances());
        if(inputFormat.classIndex()>=0)
            result.setClassIndex(result.numAttributes()-1);
        return result;	
    }

    @Override
    public Instances process(Instances inst) throws Exception {
        Instances output=determineOutputFormat(inst);
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
    public void setOptions(String[] options) throws Exception {
        String maxLagString=Utils.getOption('L', options);
        if (maxLagString.length() != 0)
            this.maxLag = Integer.parseInt(maxLagString);
        else
            this.maxLag = DEFAULT_MAXLAG;
    }

    public static void main(String[] args){





    }
	
	
}
