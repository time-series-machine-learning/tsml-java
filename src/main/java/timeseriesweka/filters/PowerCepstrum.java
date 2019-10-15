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

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

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
        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int i=0;i<length;i++){
            name = "PowerSpectrum_"+i;
            atts.add(new Attribute(name));
        }

        if(inputFormat.classIndex()>=0){	//Classification set, set class
            //Get the class values as a fast vector
            Attribute target =inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals=new ArrayList<>(target.numValues());
            for(int i=0;i<target.numValues();i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
        }

        Instances result = new Instances("Cepstrum"+inputFormat.relationName(),atts,inputFormat.numInstances());
        if(inputFormat.classIndex()>=0)
            result.setClassIndex(result.numAttributes()-1);

        return result;
    }

    @Override
    public Instances process(Instances instances) throws Exception {
//Find power spectrum                
        Instances output=super.process(instances);
//Take logs
        logDataSet(output);
//Take Inverse FFT of logged Spectrum.
        for(int i=0;i<output.numInstances();i++){
//Get out values, store in a complex array   
            Instance next=output.instance(i);
            double[] ar=next.toDoubleArray();
//Have to pad
            int n = (int)MathsPower2.roundPow2(ar.length-1);
            if(n<ar.length-1)
                n*=2;
            FFT.Complex[] complex=new FFT.Complex[n];
            for(int j=0;j<ar.length-1;j++)
                complex[j]=new FFT.Complex(ar[j],0);
            for(int j=ar.length-1;j<n;j++)
                complex[j]=new FFT.Complex(0,0);


            //Take inverse FFT
            inverseFFT(complex,complex.length);
//Square the terms for the PowerCepstrum 
            for(int j=0;j<ar.length-1;j++)
                next.setValue(j,complex[j].real*complex[j].real+complex[j].imag*complex[j].imag);

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
