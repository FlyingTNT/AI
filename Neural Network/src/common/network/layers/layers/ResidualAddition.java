package common.network.layers.layers;

import common.network.math.NetworkMath;

public class ResidualAddition extends Layer {

	Layer residual;
	
	public ResidualAddition(Layer last, Layer residual) {
		super(last, last.outputs);
		this.residual = residual;
	}

	@Override
	public float[][] activation(float[][] input) {
		lastActivation = NetworkMath.add(lastLayer.getLastActivation(), residual.getLastActivation());
		masks = lastLayer.masks;
		return lastActivation;
	}

	@Override//VERIFIED
	public void backprop() {
		float[][] gradient = getGradient();
		clearGradients();
		
		lastLayer.reportGradient(gradient);
		residual.reportGradient(gradient);
	}

	@Override
	public String name() {
		return "Residual Addition";
	}

}
