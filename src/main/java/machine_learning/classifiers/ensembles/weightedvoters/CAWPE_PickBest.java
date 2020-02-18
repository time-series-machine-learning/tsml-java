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
package machine_learning.classifiers.ensembles.weightedvoters;

import machine_learning.classifiers.ensembles.voting.BestIndividualTrain;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import machine_learning.classifiers.ensembles.CAWPE;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPE_PickBest extends CAWPE {
    public CAWPE_PickBest() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleName = "CAWPE_PickBest"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new BestIndividualTrain();
    }   
}
