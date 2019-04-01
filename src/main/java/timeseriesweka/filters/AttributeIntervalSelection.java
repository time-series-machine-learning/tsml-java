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
import weka.core.FastVector;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author ajb
 */
public class AttributeIntervalSelection extends SimpleBatchFilter{
    int start=0;
    int end=1;
    boolean trained=false;
    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
    //Check all attributes are real valued, otherwise throw exception
        for(int i=0;i<inputFormat.numAttributes();i++)
            if(inputFormat.classIndex()!=i)
                if(!inputFormat.attribute(i).isNumeric())
                        throw new Exception("Non numeric attribute not allowed in "+this.getClass().getName());
    //Create new attributes from start to end (inclusive)
        if(end-start+1>inputFormat.numAttributes()-1)
                        throw new Exception(" Too many attributes in "+this.getClass().getName()+"\n start ="+start+" end = "+end+" num Atts ="+(inputFormat.numAttributes()-1));
    //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int i=start;i<=end;i++){
            name = "Interval_"+i;
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
        Instances result = new Instances("Interval"+inputFormat.relationName(),atts,inputFormat.numInstances());
        if(inputFormat.classIndex()>=0){
                result.setClassIndex(result.numAttributes()-1);
        }
        return result;
        
        
        
    }

    @Override
    public Instances process(Instances instances){
        if(!trained){
//Find the optimal interval start and end position
            findBestInterval(instances);
            trained=true;
        }
        return instances;
    }
    private void findBestInterval(Instances instances){
        start=0;
        end=instances.numAttributes()-2;
    }

}
