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
package weka_uea.classifiers.ensembles.stackers;

import weka_uea.classifiers.ensembles.voting.stacking.StackingOnDists;
import weka_uea.classifiers.ensembles.weightings.EqualWeighting;
import weka_uea.classifiers.ensembles.CAWPE;
import weka_uea.classifiers.MultiResponseModelTrees;

/**
 * Stacking with multi-response model trees. M5 is used to induce the
 * model trees at the meta level. Dzeroski and Zenko (2004)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SMM5 extends CAWPE {
    public SMM5() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "SMM5"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new StackingOnDists(new MultiResponseModelTrees());
    }  
}
