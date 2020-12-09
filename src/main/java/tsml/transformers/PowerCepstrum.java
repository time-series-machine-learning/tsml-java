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
package tsml.transformers;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

import tsml.data_containers.TimeSeriesInstance;

/**
 *
 * @author ajb
 */
public class PowerCepstrum extends PowerSpectrum{

    public PowerCepstrum(){}


    @Override
    public Instances determineOutputFormat(Instances inputFormat) {

        //Set up instances size and format.
        int length=(fftTransformer.findLength(inputFormat));
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
    public Instance transform(Instance inst){
        Instance out = super.transform(inst);

        //log dataset
        for(int j=0;j<out.numAttributes();j++){
            if(j!=out.classIndex())
                out.setValue(j,Math.log(out.value(j)));
        }

        double[] ar=out.toDoubleArray();
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
            out.setValue(j,complex[j].real*complex[j].real+complex[j].imag*complex[j].imag);

        return out;
    }

    @Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        TimeSeriesInstance f_inst = super.transform(inst);

        double[][] values = f_inst.toValueArray();

        //log all the output/
        for(int i=0; i<values.length;i++){

            int length = values[i].length;

            for(int j=0; j<length; j++)
                values[i][j] = Math.log(values[i][j]);
            
            //Have to pad
            int n = (int)MathsPower2.roundPow2(length);
            if(n<length)
                n*=2;
            FFT.Complex[] complex=new FFT.Complex[n];
            for(int j=0;j<length;j++)
                complex[j]=new FFT.Complex(values[i][j],0);
            for(int j=length;j<n;j++)
                complex[j]=new FFT.Complex(0,0);

            //Take inverse FFT
            inverseFFT(complex,complex.length);
            //Square the terms for the PowerCepstrum 
            for(int j=0; j<length; j++)
                values[i][j] = complex[j].real*complex[j].real+complex[j].imag*complex[j].imag;
        }

        return new TimeSeriesInstance(values, inst.getLabelIndex(), inst.getClassLabels());
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
