// Poker Dice Distribution
package statistics.distributions;

public class PokerDiceDistribution extends Distribution{
	final static int c = 7776;

	public PokerDiceDistribution(){
		setParameters(0, 6, 1, DISCRETE);
	}

	public double getDensity(double x){
		double d = 0;
		int i = (int)x;
		switch(i){
		case 0:
			d = 720.0 / c;
			break;
		case 1:
			d = 3600.0 / c;
			break;
		case 2:
			d = 1800.0 / c;
			break;
		case 3:
			d = 1200.0 / c;
			break;
		case 4:
			d = 300.0 / c;
			break;
		case 5:
			d = 150.0 / c;
			break;
		case 6:
			d = 6.0 / c;
			break;
		}
		return d;
	}

	public String name(){
		return "Poker Dice Distribution";
	}
}

