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
package machine_learning.classifiers.ensembles.voting.stacking;

import weka.classifiers.Classifier;
import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * Stacking on dists, dists multiplied by the max of the probabilities, and the entropies of the dists
 * 
 * From section 3.2 of 
 * 
 (
    @article{dvzeroski2004combining,
    title={Is combining classifiers with stacking better than selecting the best one?},
    author={D{\v{z}}eroski, Saso and {\v{Z}}enko, Bernard},
    journal={Machine learning},
    volume={54},
    number={3},
    pages={255--273},
    year={2004},
    publisher={Springer}
 } 
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class StackingOnExtendedSetOfFeatures extends AbstractStacking {

    public StackingOnExtendedSetOfFeatures(Classifier classifier) {
        super(classifier);
    }
    
    public StackingOnExtendedSetOfFeatures(Classifier classifier, int numClasses) {
        super(classifier, numClasses);
    }
    
    @Override
    protected void setNumOutputAttributes(EnsembleModule[] modules) {
        this.numOutputAtts = modules.length*(numClasses*2 + 1) + 1; //each dist twice and their entopies + class val
    }
    
    @Override
    protected Instance buildInst(double[][] dists, Double classVal) {
        double[] instData = new double[numOutputAtts];
        
        int i = 0;
        for (int m = 0; m < dists.length; m++) {
            for (int c = 0; c < numClasses; c++) 
                instData[i++] = dists[m][c];
            
            double maxProbability = utilities.GenericTools.max(dists[m]);
            for (int c = 0; c < numClasses; c++) 
                instData[i++] = dists[m][c] * maxProbability;
            
            double entropy = 0;
            for (int c = 0; c < numClasses; c++) 
                entropy -= dists[m][c] * (Math.log(dists[m][c]) / Math.log(2)); //change of base formula
            instData[i++] = entropy;
        }
        
        assert(i == numOutputAtts-1);
        
        if (classVal != null)
            instData[i] = classVal; 
        //else irrelevent 
        
        instsHeader.add(new DenseInstance(1.0, instData));
        return instsHeader.remove(0);
    }
    
}
