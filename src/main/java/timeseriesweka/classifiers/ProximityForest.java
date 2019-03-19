/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package timeseriesweka.classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

/**
 * 
 * An in-progress wrapper class for the java Proximity Forest implementation: 
 *  
 * Github code:   https://github.com/fpetitjean/ProximityForest
 * 
 * @article{DBLP:journals/corr/abs-1808-10594,
 *   author    = {Benjamin Lucas and
 *                Ahmed Shifaz and
 *                Charlotte Pelletier and
 *                Lachlan O'Neill and
 *                Nayyar A. Zaidi and
 *                Bart Goethals and
 *                Fran{\c{c}}ois Petitjean and
 *                Geoffrey I. Webb},
 *   title     = {Proximity Forest: An effective and scalable distance-based classifier
 *                for time series},
 *   journal   = {CoRR},
 *   volume    = {abs/1808.10594},
 *   year      = {2018},
 *   url       = {http://arxiv.org/abs/1808.10594},
 *   archivePrefix = {arXiv},
 *   eprint    = {1808.10594},
 *   timestamp = {Mon, 03 Sep 2018 13:36:40 +0200},
 *   biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-10594},
 *   bibsource = {dblp computer science bibliography, https://dblp.org}
 * }
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ProximityForest extends AbstractClassifier {

    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public double[] distributionForInstance(Instance inst) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
