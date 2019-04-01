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
package utilities;

import utilities.InstanceTools;
import weka.core.Instances;

/**
 * Base class to create folds from a dataset in a reproducable way. 
 * This class can be subtyped to allow for dataset specific cross validation.
 * Examples include leave one person out (e.g. EpilepsyX) or leave one bottle out
 * (e.g. EthanolLevel)
 *
 * @author ajb
 */
public class FoldCreator {
    double prop=0.3;
    protected boolean deleteFirstAttribute=false;//Remove an index
    public void deleteFirstAtt(boolean b){
        deleteFirstAttribute=b;
    }
    public FoldCreator(){
        
    }
    public FoldCreator(double p){
        prop=p;
    }
    public void setProp(double p){
        prop=p;
    }
    public Instances[] createSplit(Instances data, int fold) throws Exception{
        Instances[] split= InstanceTools.resampleInstances(data, fold, prop);
        if(deleteFirstAttribute){
            split[0].deleteAttributeAt(0);
            split[1].deleteAttributeAt(0);
        }
        return split;
    } 
    
}
