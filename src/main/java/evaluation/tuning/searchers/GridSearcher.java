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
package evaluation.tuning.searchers;

import evaluation.tuning.ParameterSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class GridSearcher extends ParameterSearcher {
    
    @Override
    public Iterator<ParameterSet> iterator() {
        return new GridSearchIterator();
    }
    
    public class GridSearchIterator implements Iterator<ParameterSet> {
        
        int numParas;
        String[] keys;
        int[] sizes;
        int[] currentInds;
        
        public GridSearchIterator() {
            numParas = space.numParas();
            
            currentInds = new int[numParas]; //keep init to all 0's 
            sizes = new int[numParas];
            keys = new String[numParas];
            
            int i = 0;
            for (Map.Entry<String, List<String>> entry : space.parameterLists.entrySet()) {
                keys[i] = entry.getKey();
                sizes[i] = entry.getValue().size();
                i++;
            }
        }
        
        @Override
        public boolean hasNext() {
            //we havnt reached the end of the outer-most parameter list 
            return currentInds[0] < sizes[0]; 
        }

        @Override
        public ParameterSet next() {
            
            ///////////build the current parameter set 
            ParameterSet pset = new ParameterSet();
            for (int i = 0; i < keys.length; i++)
                pset.parameterSet.put(keys[i], space.parameterLists.get(keys[i]).get(currentInds[i]));
            
            ///////////increment to the next one
            //increment the inner-most parameter list
            int currentParaConsidered = numParas-1;
            currentInds[currentParaConsidered]++;
            
            //have we reached the end of this parameter list? 
            while (currentInds[currentParaConsidered] == sizes[currentParaConsidered]) {
                if (currentParaConsidered == 0)
                    //hasNext() will use this fact that the outer-most para is done to determine that there are none left
                    break;
                
                //reset this list to 0 to cycle round again
                currentInds[currentParaConsidered] = 0;
                
                //increment the next-outer parameter list
                currentInds[--currentParaConsidered]++;
            }
            
            return pset;
        }
        
    }
}
