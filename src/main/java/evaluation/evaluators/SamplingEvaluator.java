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
package evaluation.evaluators;

/**
 * Base class for evaluators that will sample the data provided to the evaluate function
 * in some way, and build the classifier at least once on some part of the data. 
 * Classifiers need not have been built on some from of data prior to using this evaluator.
 * 
 * In this sense, evaluators extending this class can be viewed as 'self contained', in
 * that no external data is needed.
 * 
 * NOTE: A typical extension of this class might be to e.g. resample the data passed 
 * into a train and test set, build on the train and predict the test. As a result of this 
 * use case, the number of predictions in the returned classifierresults object is 
 * not equal to the number of instances in the data passed. 
 * 
 * Currently, this class does not add any extra functionality, however acts as an
 * extra step in the inheritance hierarchy to distinguish 'self-contained' and 'not self-contained'
 * evaluators by the loose definition above.
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public abstract class SamplingEvaluator extends Evaluator {

    public SamplingEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed, cloneData, setClassMissing);
    }

}
