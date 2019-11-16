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
 *
 * @author James Large james.large@uea.ac.uk
 */
public class StackingOnDists extends AbstractStacking {

    public StackingOnDists(Classifier classifier) {
        super(classifier);
    }
    
    public StackingOnDists(Classifier classifier, int numClasses) {
        super(classifier, numClasses);
    }
    
    @Override
    protected void setNumOutputAttributes(EnsembleModule[] modules) {
        this.numOutputAtts = modules.length*numClasses + 1; //each dist + class val
    }
    
    @Override
    protected Instance buildInst(double[][] dists, Double classVal) {
        double[] instData = new double[numOutputAtts];
        
        int i = 0;
        for (int m = 0; m < dists.length; m++) 
            for (int c = 0; c < numClasses; c++) 
                instData[i++] = dists[m][c];
        
        assert(i == numOutputAtts-1);
        
        if (classVal != null)
            instData[i] = classVal; 
        //else irrelevent 
        
        instsHeader.add(new DenseInstance(1.0, instData));
        return instsHeader.remove(0);
    }
    
}
