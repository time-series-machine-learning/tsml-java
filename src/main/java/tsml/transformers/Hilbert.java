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

import java.io.FileReader;

import utilities.InstanceTools;
import weka.core.*;
import weka.filters.SimpleBatchFilter;

/*
     * copyright: Anthony Bagnall
	 * @author Aaron Bostrom
 * */
public class Hilbert implements Transformer {


	public Instances determineOutputFormat(Instances inputFormat) throws IllegalArgumentException{
		FastVector<Attribute> atts=new FastVector<>();

		for(int i=0;i<inputFormat.numAttributes()-1;i++)
		{
		//Add to attribute list                          
		String name = "Hilbert"+i;
		atts.addElement(new Attribute(name));
		}
		//Get the class values as a fast vector			
		Attribute target =inputFormat.attribute(inputFormat.classIndex());

		FastVector<String> vals=new FastVector<>(target.numValues());
		for(int i=0;i<target.numValues();i++)
				vals.addElement(target.value(i));
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(),vals));
		Instances result = new Instances("Hilbert"+inputFormat.relationName(),atts,inputFormat.numInstances());
		if(inputFormat.classIndex()>=0){
				result.setClassIndex(result.numAttributes()-1);
		}

		return result;
	}

	@Override
	public void fit(Instances data) {
		// TODO Auto-generated method stub

	}

	@Override
	public Instances transform(Instances data) {
		//for k=1 to n: f_k = sum_{i=1}^n f_i cos[(k-1)*(\pi/n)*(i-1/2)] 
		//Assumes the class attribute is in the last one for simplicity            
		Instances result = determineOutputFormat(data);        
		for(Instance inst : data) 
			result.add(transform(inst));

		return result;
	}
	
	@Override
	public Instance transform(Instance inst) {
		int n=inst.numAttributes()-1;
		Instance newInst= new DenseInstance(inst.numAttributes());
		for(int k=0;k<n;k++){
			double fk=0;
			for(int i=0;i<n;i++){
				if(i!=k)
					fk+=inst.value(i)/(k-i);
			}
			newInst.setValue(k, fk);
		}
		newInst.setValue(inst.classIndex(), inst.classValue());
		return newInst;
	}

	public static void main(String[] args){
		final double[][] t1 = {{0, Math.PI, Math.PI*2},{ Math.PI * 0.5, Math.PI * 1.5, Math.PI*2.5}};
        final Instances train = InstanceTools.toWekaInstances(t1);

        Hilbert hilbertTransform= new Hilbert();
        Instances out_train = hilbertTransform.fitTransform(train);
        System.out.println(out_train);
	}



}
