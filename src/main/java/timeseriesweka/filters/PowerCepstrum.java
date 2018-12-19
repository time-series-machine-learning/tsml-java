/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author ajb
 */
public class PowerCepstrum extends PowerSpectrum{
    
    public PowerCepstrum(){
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		
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
		Instances result = new Instances("Cepstrum"+inputFormat.relationName(),atts,inputFormat.numInstances());
		if(inputFormat.classIndex()>=0)
			result.setClassIndex(result.numAttributes()-1);
                
      		return result;
	}

    @Override
    public Instances process(Instances instances) throws Exception {
//Find power spectrum                
		Instances output=determineOutputFormat(instances);
                 output=super.process(instances);
//Take logs
                logDataSet(output);
//Take Inverse FFT of logged Spectrum.
               for(int i=0;i<output.numInstances();i++){
//Get out values, store in a complex array   
                   Instance next=output.instance(i);
                   double[] ar=next.toDoubleArray();
                   FFT.Complex[] complex=new FFT.Complex[ar.length-1];
                   for(int j=0;j<ar.length-1;j++)
                       complex[i]=new FFT.Complex(ar[i],0);
//Take inverse FFT
                   inverseFFT(complex,complex.length);
//Square the terms for the PowerCepstrum 
                   for(int j=0;j<complex.length;j++)
                       next.setValue(j,complex[i].real*complex[i].real+complex[i].imag*complex[i].imag);
                       
               } 
                
                return output;

    }
    public void logDataSet(Instances out ){
        for(int i=0;i<out.numInstances();i++){
            Instance ins=out.instance(i);
            for(int j=0;j<ins.numAttributes();j++){
                if(j!=ins.classIndex())
                    ins.setValue(j,Math.log(ins.value(j)));
            }
        }
            
        
    }
}
