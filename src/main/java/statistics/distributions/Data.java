package statistics.distributions;
import java.util.*;

/**A simple implementation of a data distribution*/
public class Data{
	//Variables
	private Vector values = new Vector();
	private int size;
	private double value, mean, meanSquare, mode;
	private String name;

	/**This general constructor creates a new data with a prescribed name.*/
	public Data(String n){
		setName(n);
	}

	/**This default constructor creates a new data with the name "X"*/
	public Data(){
		this("X");
	}

	/**This method adds a new number to the data set and re-compute the mean, mean square,
	minimum and maximum values, and order statistics*/
	public void setValue(double x){
		double a, b;
		value = x;
		boolean notInserted = true;
		//Add the value to the data set
		for (int i = 0; i < size - 1; i++){
			a = ((Double)values.elementAt(i)).doubleValue();
			b = ((Double)values.elementAt(i + 1)).doubleValue();
			if ((a <= x) & (x >= b)){
				values.insertElementAt(new Double(x), i + 1);
				notInserted = false;
			}
		}
		if (notInserted) values.insertElementAt(new Double(x), 0);
		//Re-compute mean and mean square
		mean = ((double)(size - 1) / size) * mean + value / size;
		meanSquare = ((double)(size - 1) / size) * meanSquare + value * value / size;
	}

	/**Get the current value of the data set*/
	public double getValue(){
		return value;
	}

	/**This method returns the i'th value of the data set.*/
	public double getValue(int i){
		return ((Double)values.elementAt(i)).doubleValue();
	}

	/**Get the mean*/
	public double getMean(){
		return mean;
	}

	/**Get the population variance*/
	public double getPVariance(){
		double var = meanSquare - mean * mean;
		if (var < 0) var = 0;
		return var;
	}

	/**Get the population standard deviation*/
	public double getPSD(){
		return Math.sqrt(getPVariance());
	}

	/**Get the sample variance of the data set*/
	public double getVariance(){
		return ((double)size / (size - 1)) * getPVariance();
	}

	/**Get the sample standard deviation of the data set*/
	public double getSD(){
		return Math.sqrt(getVariance());
	}

	/**Get the minimum value of the data set*/
	public double getMinValue(){
		return getValue(0);
	}

	/**Get the maximum value of the data set*/
	public double getMaxValue(){
		return getValue(size - 1);
	}

	/**Reset the data set*/
	public void reset(){
		values.removeAllElements();
		size = 0;
	}

	/**Get the number of pointCount in the data set*/
	public int getSize(){
		return size;
	}

	/**Get the name of the data set*/
	public void setName(String name){
		this.name = name;
	}

	/**Set the name of the data set*/
	public String getName(){
		return name;
	}
}
