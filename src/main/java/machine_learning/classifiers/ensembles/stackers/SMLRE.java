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
package machine_learning.classifiers.ensembles.stackers;

import machine_learning.classifiers.ensembles.voting.stacking.StackingOnExtendedSetOfFeatures;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import machine_learning.classifiers.ensembles.CAWPE;
import machine_learning.classifiers.MultiLinearRegression;

/**
 * Stacking with MLR and an extended set of meta-level attributes, Dzeroski and Zenko (2004)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SMLRE extends CAWPE{
    public SMLRE() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleName = "SMLRE"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new StackingOnExtendedSetOfFeatures(new MultiLinearRegression());
    }   
}
