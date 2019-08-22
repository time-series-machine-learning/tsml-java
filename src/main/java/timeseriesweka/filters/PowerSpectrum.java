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

import experiments.data.DatasetLoading;
import fileIO.OutFile;
import java.io.FileReader;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/** 
  <!-- globalinfo-start -->
* Implementation of power spectrum function as a Weka SimpleBatchFilter
* Series to series transform independent of class value
*  <!-- globalinfo-end -->
<!-- options-start -->
 * Valid options are: <p/>
 * TO DO
 <!-- options-end -->
 *
 * 
 * author: Anthony Bagnall circa 2008.
 Reviewed and tidied up 2019
 */
public class PowerSpectrum extends FFT {
    boolean log=false;
    FFT fftFilter;	
    public void takeLogs(boolean x){log=x;}

    public PowerSpectrum(){
        fftFilter=new FFT();
        fftFilter.useDFT();
    }

    public void useFFT(){
        fftFilter.useFFT();
    }
@Override
    protected Instances determineOutputFormat(Instances inputFormat)
    throws Exception {
        //Set up instances size and format.
        int length=(fftFilter.findLength(inputFormat));
        length/=2;
        FastVector atts=new FastVector();
        String name;
        for(int i=0;i<length;i++){
                name = "PowerSpectrum_"+i;
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
        Instances result = new Instances("PowerSpectrum"+inputFormat.relationName(),atts,inputFormat.numInstances());
        if(inputFormat.classIndex()>=0)
                result.setClassIndex(result.numAttributes()-1);

        return result;
    }
    @Override
    public Instances process(Instances instances) throws Exception {
//Get the FFT
        //For each data, first extract the relevant data
        Instances output=determineOutputFormat(instances);
        Instances fft=fftFilter.process(instances);
        int length=fft.numAttributes();
        if(instances.classIndex()>=0)
                length--;
        length/=2;

        if(log)
        {
            double l1;
            for(int i=0;i<fft.numInstances();i++){			
                Instance f=fft.instance(i);
                Instance inst=new DenseInstance(length+1);
                for(int j=0;j<length;j++){
                    l1= Math.sqrt(f.value(j*2)*f.value(j*2)+f.value(j*2+1)*f.value(j*2+1));
                        inst.setValue(j,Math.log(l1));
                }
//Set class value.
                //Set class value.
                if(output.classIndex()>=0)
                    inst.setValue(length, fft.instance(i).classValue());
                output.add(inst);
            }

        }
        else{
            for(int i=0;i<fft.numInstances();i++){			
                    Instance f=fft.instance(i);
                    Instance inst=new DenseInstance(length+1);
                    for(int j=0;j<length;j++){
                                    inst.setValue(j, Math.sqrt(f.value(j*2)*f.value(j*2)+f.value(j*2+1)*f.value(j*2+1)));
                    }
    //Set class value.
                    //Set class value.
                    if(output.classIndex()>=0)
                            inst.setValue(length, fft.instance(i).classValue());
                    output.add(inst);
            }
        }
        return output;		
    }
    public static void waferTest(){
/*		Instances a=WekaMethods.loadDataThrowable("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TRAIN");
            Instances b=WekaMethods.loadDataThrowable("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TEST");
            PowerSpectrum ps=new PowerSpectrum();
            try{
            Instances c=ps.process(a);
            Instances d=ps.process(b);
            OutFile of = new OutFile("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TRAIN_PS.arff");
            OutFile of2 = new OutFile("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TEST_PS.arff");
            of.writeString(c.toString());
            of2.writeString(d.toString());
            }catch(Exception e){
                    System.out.println(" Exception ="+e);
            }
*/	}
    /* Transform by the  built in filter*/
    public static double[] powerSpectrum(double[] d){

//Check power of 2            
        if(((d.length)&(d.length-1))!=0)    //Not a power of 2
            return null;
        FFT.Complex[] c=new FFT.Complex[d.length];
        for(int j=0;j<d.length;j++){
             c[j]=new FFT.Complex(d[j],0.0);
        }
        FFT f=new FFT();
        f.fft(c,c.length);
        double[] ps=new double[c.length];
        for(int i=0;i<c.length;i++)
            ps[i]=c[i].getReal()*c[i].getReal()+c[i].getImag()*c[i].getImag();
        return ps;
    }

    public static Instances loadData(String fullPath)
    {
            Instances d=null;
            FileReader r;
            int nosAtts;
            try{		
                    r= new FileReader(fullPath+".arff"); 
                    d = new Instances(r); 
                    d.setClassIndex(d.numAttributes()-1);
            }
            catch(Exception e)
            {
                    System.out.println("Unable to load data on path "+fullPath+" Exception thrown ="+e);
                    e.printStackTrace();
                    System.exit(0);
            }
            return d;
    }        
    public static void matlabComparison(){

//MATLAB Output generated by            
        // Power of 2: use FFT
//Create set of instances with 16 attributes, with values
// Case 1:           All Zeros
// Case 2:           1,2,...16
// Case 3:           -8,-7, -6,...,0,1,...7
//Case 4:           0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1              
/*            PowerSpectrum ps=new PowerSpectrum();

        Instances test1=ClassifierTools.loadDataThrowable("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\FFT_test1");
        Instances test2=ClassifierTools.loadDataThrowable("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\FFT_test2");
        Instances t2;
     try{
          t2=ps.process(test1);
            System.out.println(" TEST 1 PS ="+t2);
            t2=ps.process(test2);
            System.out.println(" TEST 2 PS ="+t2);


        }catch(Exception e){
            System.out.println(" Errrrrrr = "+e);
            e.printStackTrace();
            System.exit(0);
        }    
*/          
// Not a power of 2: use padding

// Not a power of 2: use truncate

// Not a power of 2: use DFT

    }


    public static void main(String[] args) {
        String problemPath = "E:/TSCProblems/";
        String resultsPath="E:/Temp/";
        String datasetName="ItalyPowerDemand";
        Instances train =DatasetLoading.loadDataNullable("E:/TSCProblems/"+datasetName+"/"+datasetName+"_TRAIN");
        PowerSpectrum ps= new PowerSpectrum();
        try {
            Instances trans=ps.process(train);
            OutFile out = new OutFile(resultsPath+datasetName+"PS_JAVA.csv");
            out.writeLine(datasetName);
            for(Instance ins: trans){
                double[] d=ins.toDoubleArray();
                for(int j=0;j<d.length;j++){
                    if(j!=trans.classIndex())
                        out.writeString(d[j]+",");
                }
                out.writeString("\n");
            }
        } catch (Exception ex) {
            System.out.println("ERROR IN DEMO");
            Logger.getLogger(ACF.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
