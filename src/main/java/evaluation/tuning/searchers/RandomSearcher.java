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
import evaluation.tuning.searchers.GridSearcher.GridSearchIterator;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class RandomSearcher extends ParameterSearcher {
    
    int numParaSetsToSample;

    public RandomSearcher() { 
        numParaSetsToSample = 1000;
    }
    public RandomSearcher(int numParasToSample) { 
        this.numParaSetsToSample = numParasToSample;
    }
    
    public int getNumParaSetsToSearch() {
        return numParaSetsToSample;
    }

    public void setNumParaSetsToSearch(int numParaSetsToSearch) {
        this.numParaSetsToSample = numParaSetsToSearch;
    }

    @Override
    public Iterator<ParameterSet> iterator() {
        if (numParaSetsToSample < space.numUniqueParameterSets())
            return new RandomSearchIterator();
        else {
            System.out.println("Warning: tryign to randomly sample a space more times than there are unique values in the space, just using a GridSearch");
            return new GridSearcher().iterator();
        } 
    }
    
    public class RandomSearchIterator implements Iterator<ParameterSet> {
        Random rng;
        int numParaSetsSampled;
        
        int numParas;
        String[] keys;
        int[] sizes;
        Set<String> vistedParas = new TreeSet<>();
        
        public RandomSearchIterator() {
            numParaSetsSampled = 0;
            numParas = space.numParas();
            
            rng = new Random(seed);
            
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
            return numParaSetsSampled < numParaSetsToSample; 
        }

        public int[] sampleParas(int secondSeed) {
            Random trng = new Random(secondSeed);
            int[] set = new int[numParas];
            boolean isNewSet = true;
            
            do {
                set[0] = trng.nextInt(sizes[0]);
                for (int i = 1; i < numParas; i++)
                    set[i] = trng.nextInt(sizes[i]); 
                
                isNewSet = vistedParas.add(ParameterSet.toFileNameString(set));
            } 
            while (!isNewSet);
            
            return set;
        }
        
        @Override
        public ParameterSet next() {
            
            //so, to replicate this parameter set (in e.g a checkpointed/para split scenario)
            //init the searcher with same fold-seed, and iterate through same number of times
            int[] psetInds = sampleParas(rng.nextInt());
            
            ParameterSet pset = new ParameterSet();
            for (int i = 0; i < keys.length; i++)
                pset.parameterSet.put(keys[i], space.parameterLists.get(keys[i]).get(psetInds[i]));
            
            numParaSetsSampled++;
            
            return pset;
        }
        
    }
}
