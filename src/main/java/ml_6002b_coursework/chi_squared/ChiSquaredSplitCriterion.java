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

/*
 *    EntropyBasedSplitCrit.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package ml_6002b_coursework.chi_squared;

import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.SplitCriterion;

/**
 * class for computing chi squared splitting criteria
 */
public class ChiSquaredSplitCriterion
  extends SplitCriterion {

  /** for serialization */
  private static final long serialVersionUID = -2618691439791653056L;
  /**
   * Computes result of splitting criterion for given distribution.
   *
   * @return value of splitting criterion. 0 by default
   */
  @Override
  public double splitCritValue(Distribution bags){
// HERE WORK OUT THE GINI FOR SPLITTING INTO THE DISTRIBUTION PASSED


    return 0;
  }


  @Override
  public String getRevision() {
    return null;
  }
}

