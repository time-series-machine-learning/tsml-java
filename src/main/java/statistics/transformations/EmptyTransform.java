/*
 * Created on Jan 30, 2006
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package transformations;

import weka.core.Instances;

/**
 * @author ajb
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class EmptyTransform extends Transformations {

	public Instances transform(Instances data) {
		return data;
	}

	public Instances invert(Instances data) {
		return data;
	}
	public Instances staticTransform(Instances data) {
		return data;
	}
	public double[] invertPredictedResponse(double[] d) {
		return d;
	}

	public static void main(String[] args) {
	}
}
