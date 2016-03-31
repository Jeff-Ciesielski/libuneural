#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <uneural.h>
//#define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

static fix16_t uneural_sigmoid_deriv(fix16_t v)
{
	fix16_t deriv = 0;
	deriv = fix16_ssub(F16(1), v);
	deriv = fix16_smul(v, deriv);

	return deriv;

}

static fix16_t uneural_tanh_deriv(fix16_t v)
{
	fix16_t deriv = 0;
	deriv = fix16_sq(v);
	deriv = fix16_ssub(F16(1), deriv);
	return deriv;
}

static fix16_t uneural_relu_deriv(fix16_t v)
{

	if (v <= 0)
		return F16(0);
	else
		return F16(1);
}

static fix16_t uneural_leaky_relu_deriv(fix16_t v)
{

	if (v <= 0)
		return F16(.01);
	else
		return F16(1);

}

static fix16_t uneural_random_weight(void)
{
	fix16_t temp;

	temp = fix16_from_float((float)rand() / (float)RAND_MAX);

	return temp;
}

uint16_t uneural_network_largest_layer_size(struct uneural_network *n)
{
	uint16_t max_layer_size = 0;

	for (struct uneural_layer *l = n->input; l != NULL; l = l->next) {
		if (l->num_neurons > max_layer_size)
			max_layer_size = l->num_neurons;
	}

	return max_layer_size;
}

ssize_t uneural_network_get_training_scratch_size(struct uneural_network *n)
{

	if (n == NULL) {
		return -NULL_ARG;
	}

	uint16_t max_layer_size = uneural_network_largest_layer_size(n);

	return (max_layer_size * max_layer_size) * 8 * sizeof(fix16_t);
}

void print_network_neurons(struct uneural_network *n)
{

	for (struct uneural_layer *l = n->input; l != NULL; l = l->next) {

		if (l == n->input) {
			DEBUG_PRINT("Input Layer\n");
			for (int i = 0; i < l->num_neurons; i++) {
				DEBUG_PRINT("-INPUT\n");
			}
			continue;
		} else if (l == n->output) {
			DEBUG_PRINT("Output Layer\n");
		} else {
			DEBUG_PRINT("Hidden Layer\n");
		}
		for (int i = 0; i < l->num_neurons; i++) {

			switch (*l->neurons[i].n_type) {
			case NEURON_TYPE_SIGMOID:
				DEBUG_PRINT("-SIGMOID\n");
				break;
			case NEURON_TYPE_TANH:
				DEBUG_PRINT("-TANH\n");
				break;
			case NEURON_TYPE_RELU:
				DEBUG_PRINT("-RELU\n");
				break;
			case NEURON_TYPE_LEAKY_RELU:
				DEBUG_PRINT("-LEAKYRELU\n");
				break;
			}

		}

	}

}

/* TODO: Add size check to inputs and outputs */
int uneural_network_backprop(struct uneural_network *n,
                             const fix16_t *input,
                             const fix16_t *expected_output,
                             fix16_t training_rate,
                             fix16_t *scratch,
                             fix16_t *output_error)
{

	if (n == NULL || input == NULL ||
	    expected_output == NULL || scratch == NULL) {
		return -NULL_ARG;
	}

	int res = uneural_activate_network(n,
					   input,
					   NULL);

	if (res) {
		DEBUG_PRINT("Error activating network\n");
		return res;
	}

	uint32_t max_layer_size = uneural_network_largest_layer_size(n);
	uint32_t step_size = (max_layer_size * max_layer_size) * sizeof(fix16_t);

	fix16_t *l1_output = scratch;
	fix16_t *l1_error = l1_output + step_size;
	fix16_t *l1_delta = l1_error + step_size;
	fix16_t *l1_start_weight = l1_delta + step_size;
	fix16_t *l2_output = l1_start_weight + step_size;
	fix16_t *l2_error = l2_output + step_size;
	fix16_t *l2_delta = l2_error + step_size;
	fix16_t *l2_start_weight = l2_delta + step_size;
	//print_network_neurons(n);

	for (struct uneural_layer *l = n->output; l != n->input; l = l->prev) {
		struct uneural_layer *l_p = l->prev;
		struct uneural_layer *l_n = l->next;

		if (l == n->output) {
			DEBUG_PRINT("Output Layer\n");
			for (int i = 0; i < l->num_neurons; i++) {

				/* Copy in the initial weights */
				for (int j = 0; j < l_p->num_neurons; j++) {
					l2_start_weight[l->num_neurons * i + j] = l->neurons[i].weights[j];
					l1_output[j] = l_p->neurons[j].output;
					DEBUG_PRINT("[%d|%d]w: %f\n", i, j,
						    fix16_to_float(l2_start_weight[l->num_neurons * i + j]));

				}

				l2_output[i] = l->neurons[i].output;
				l2_error[i] = fix16_ssub(expected_output[i],
							 l->neurons[i].output);

				output_error[i] = l2_error[i];

				fix16_t deriv = 0;

				switch (*l->neurons[i].n_type) {
				case NEURON_TYPE_SIGMOID:
					deriv = uneural_sigmoid_deriv(l2_output[i]);
					break;
				case NEURON_TYPE_TANH:
					deriv = uneural_tanh_deriv(l2_output[i]);
					break;
				case NEURON_TYPE_RELU:
					deriv = uneural_relu_deriv(l2_output[i]);
					break;
				case NEURON_TYPE_LEAKY_RELU:
					deriv = uneural_leaky_relu_deriv(l2_output[i]);
					break;
				}

				l2_delta[i] = fix16_smul(l2_error[i], deriv);
				DEBUG_PRINT("[%d]o/d: %f/%f\n", i,
					    fix16_to_float(l2_output[i]),
					    fix16_to_float(l2_delta[i]));
			}

		} else {
			DEBUG_PRINT("Hidden Layer\n");
			for (int i = 0; i < l->num_neurons; i++) {
				fix16_t sum = F16(0);

				/* Copy in the initial weights */
				for (int j = 0; j < l_p->num_neurons; j++) {
					l2_start_weight[l->num_neurons * i + j] = l->neurons[i].weights[j];
				}

				/* Back propegate error from next layer */
				for (int j = 0; j < l_n->num_neurons; j++) {
					DEBUG_PRINT("%d:%d - ", i, j);

					fix16_t temp = fix16_smul(l1_delta[j],
								  l1_start_weight[l_n->num_neurons * j + i]);

					DEBUG_PRINT("d/w/e: %f/%f/%f\n",
						    fix16_to_float(l1_delta[j]),
						    fix16_to_float(l1_start_weight[l_n->num_neurons * j + i]),
						    fix16_to_float(temp));

					sum  = fix16_sadd(sum, temp);
					DEBUG_PRINT("Err Sum: %f\n", fix16_to_float(sum));
				}

				l2_output[i] = l->neurons[i].output;
				fix16_t deriv = 0;

				switch (*l->neurons[i].n_type) {
				case NEURON_TYPE_SIGMOID:
					deriv = uneural_sigmoid_deriv(l2_output[i]);
					break;
				case NEURON_TYPE_TANH:
					deriv = uneural_tanh_deriv(l2_output[i]);
					break;
				case NEURON_TYPE_RELU:
					deriv = uneural_relu_deriv(l2_output[i]);
					break;
				case NEURON_TYPE_LEAKY_RELU:
					deriv = uneural_leaky_relu_deriv(l2_output[i]);
					break;
				}
				deriv = fix16_smul(sum, deriv);

				l2_delta[i] = deriv;
				DEBUG_PRINT("l2 delta%f \n", fix16_to_float(l2_delta[i]));
				DEBUG_PRINT("--------------------\n");
			}
		}

		/* Update working layer weights */
		/* TODO: Add alpha/momentum term */
		//DEBUG_PRINT("Updating layer weights\n");
		for (int i = 0; i < l->num_neurons; i++) {
			for (int j = 0; j < l_p->num_neurons; j++) {
				fix16_t layer_adj = fix16_smul(l2_delta[i],
							       l_p->neurons[j].output);

				layer_adj = fix16_smul(layer_adj,
						       training_rate);
				DEBUG_PRINT("[%d, %d] error/adj: %f|%f\n", i, j,
					    fix16_to_float(l2_error[i]),
					    fix16_to_float(layer_adj));
				l->neurons[i].weights[j] = fix16_sadd(l->neurons[i].weights[j],
								      layer_adj);
			}
			/* Adjust the neuron's bias */
			fix16_t bias_adj = l2_delta[i];
			bias_adj = fix16_smul(bias_adj, training_rate);
			l->neurons[i].bias[0] = fix16_sadd(l->neurons[i].bias[0], bias_adj);
			DEBUG_PRINT("Bias adjust: %f\n",
				    fix16_to_float(bias_adj));

		}

		memcpy(l1_output,       l2_output, step_size);
		memcpy(l1_error,        l2_error, step_size);
		memcpy(l1_delta,        l2_delta, step_size);
		memcpy(l1_start_weight, l2_start_weight, step_size);
	}
	return 0;
}

int uneural_network_randomize_weights(struct uneural_network *n)
{
	if (n == NULL) {
		return -NULL_ARG;
	}

	if (n->storage_attached == false) {
		return -MISSING_DATA_STORAGE;
	}

	/* We skip the input layer as no bias or weight are required, it
	 * exists simply as a programming convenience */
	struct uneural_layer *l = n->input->next;

	while (l != NULL) {
		/* Make sure L has local neuron storage affixed */
		if (l->neurons == NULL) {
			return -MISSING_NEURON;
		}

		/* Walk the neurons individually and assign them random weight and
		 * bias storage */
		for(int i = 0; i < l->num_neurons; i++) {
			l->neurons[i].bias[0] = uneural_random_weight();
			for (int j = 0; j < l->prev->num_neurons; j++) {
				l->neurons[i].weights[j] = uneural_random_weight();
			}
		}
		l = l->next;
	}

	return 0;
}
