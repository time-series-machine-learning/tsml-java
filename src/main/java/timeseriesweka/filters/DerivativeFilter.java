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
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.*;


public class DerivativeFilter extends SimpleBatchFilter{

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        
        int numAttributes = inputFormat.numAttributes();
        FastVector atts = new FastVector();
        String name;
        for(int i = 0; i < numAttributes-1; i++){
            name = "Attribute_" + i;
            atts.addElement(new Attribute(name));
        }
        
        if(inputFormat.classIndex() >= 0){ //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());
//
            FastVector vals = new FastVector(target.numValues());
            for(int i = 0; i < target.numValues(); i++){
                vals.addElement(target.value(i));
            }
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("derivativeTransform_" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if(inputFormat.classIndex() >= 0){
            result.setClassIndex(result.numAttributes() - 1);
        }
        
        return result;
        
    }
    
    public Instances process(Instances data) throws Exception{

        Instances output = determineOutputFormat(data);

        // for each data, get distance to each shapelet and create new instance
        for(int i = 0; i < data.numInstances(); i++){ // for each data
            Instance toAdd = new DenseInstance(data.numAttributes());
            
            double[] rawData = data.instance(i).toDoubleArray();
            double[] derivative = getDerivative(rawData,true); // class value has now been removed - be careful!

            for(int j = 0; j < derivative.length; j++){
                toAdd.setValue(j, derivative[j]);
            }

            toAdd.setValue(derivative.length, data.instance(i).classValue());
            output.add(toAdd);
        }
        return output;
    }
    
    
    private static double[] getDerivative(double[] input, boolean classValOn){

        int classPenalty = 0;
        if(classValOn){
            classPenalty=1;
        }

        double[] derivative = new double[input.length-classPenalty];

        for(int i = 1; i < input.length-1-classPenalty;i++){ // avoids class Val if present
            derivative[i] = ((input[i]-input[i-1])+((input[i+1]-input[i-1])/2))/2;
        }

        derivative[0] = derivative[1];
        derivative[derivative.length-1] = derivative[derivative.length-2];

        return derivative;
    }



}
