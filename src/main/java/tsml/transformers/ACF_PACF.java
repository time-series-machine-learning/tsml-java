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
package tsml.transformers;

import java.util.ArrayList;

import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> Implementation of partial autocorrelation function
 * as a Weka SimpleBatchFilter Series to series transform independent of class
 * value <!-- globalinfo-end --> <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 *  -L
 *  set the max lag.
 * </pre>
 * 
 * <!-- options-end -->
 *
 * 
 * author: Anthony Bagnall 2019. Transform for RISE: It finds both the ACF and
 * PACF features and concatenates them. Will ditch AR and test if there is a
 * difference
 */

public class ACF_PACF extends PACF {

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        // Check capabilities for the filter. Can only handle real valued, no missing.
        // getCapabilities().testWithFail(inputFormat);

        seriesLength = inputFormat.numAttributes();
        if (inputFormat.classIndex() >= 0)
            seriesLength--;
        if (maxLag > seriesLength - endTerms)
            maxLag = seriesLength - endTerms;
        if (maxLag < 0)
            maxLag = inputFormat.numAttributes() - 1;
        // Set up instances size and format.
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int i = 0; i < maxLag; i++) {
            name = "ACF_" + i;
            atts.add(new Attribute(name));
        }
        for (int i = 0; i < maxLag; i++) {
            name = "PACF_" + i;
            atts.add(new Attribute(name));
        }
        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values
            Attribute target = inputFormat.attribute(inputFormat.classIndex());
            ArrayList<String> vals = new ArrayList<>(target.numValues());
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("PACF" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0)
            result.setClassIndex(result.numAttributes() - 1);
        return result;
    }

    @Override
    public Instance transform(Instance inst) {
        double[] d = InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst);
        // 2. Fit Autocorrelations, if not already set externally
        autos = ACF.fitAutoCorrelations(d, maxLag);
        // 3. Form Partials
        partials = formPartials(autos);
        // 5. Find parameters
        double[] pi = new double[maxLag];
        for (int k = 0; k < maxLag; k++) { // Set NANs to zero
            if (Double.isNaN(partials[k][k]) || Double.isInfinite(partials[k][k])) {
                pi[k] = 0;
            } else
                pi[k] = partials[k][k];
        }

        int length = autos.length + (pi.length + inst.classIndex() >= 0 ? 1 : 0); // ACF atts + PACF atts + optional
                                                                                // classvalue.

        // 6. Stuff back into new Instances.
        Instance in = new DenseInstance(length);
        // Set class value.

        if (inst.classIndex() >= 0) {
            in.setValue(length - 1, inst.classValue());
        }
        int count = 0;
        for (int k = 0; k < autos.length; k++) {
            in.setValue(count, autos[k]);
            count++;
        }
        for (int k = 0; k < pi.length; k++) {
            in.setValue(count, pi[k]);
            count++;
        }

        return in;
    }

    public static double[][] formPartials(double[] r) {
        // Using the Durban-Leverson
        int p = r.length;
        double[][] phi = new double[p][p];
        double numerator, denominator;
        phi[0][0] = r[0];

        for (int k = 1; k < p; k++) {
            // Find diagonal k,k
            // Naive implementation, should be able to do with running sums
            numerator = r[k];
            for (int i = 0; i < k; i++)
                numerator -= phi[i][k - 1] * r[k - 1 - i];
            denominator = 1;
            for (int i = 0; i < k; i++)
                denominator -= phi[k - 1 - i][k - 1] * r[k - 1 - i];
            phi[k][k] = numerator / denominator;
            // Find terms 1 to k-1
            for (int i = 0; i < k; i++)
                phi[i][k] = phi[i][k - 1] - phi[k][k] * phi[k - 1 - i][k - 1];
        }
        return phi;
    }

    public double[][] getPartials() {
        return partials;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 2:2019 $");
    }

    public void setOptions(String[] options) throws Exception {
        String maxLagString = Utils.getOption('L', options);
        if (maxLagString.length() != 0)
            this.maxLag = Integer.parseInt(maxLagString);
        else
            this.maxLag = DEFAULT_MAXLAG;
    }

    public static void main(String[] args) {

    }

}
