#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <sys/types.h>

#include <uneural.h>

DECLARE_UNEURAL_LAYER(input_layer, 3);
DECLARE_UNEURAL_LAYER(hidden_layer, 4);
DECLARE_UNEURAL_LAYER(output_layer, 1);

#define ACTIVATION_TYPE NEURON_TYPE_SIGMOID
#define LEARNING_RATE F16(.1)
#define TRAINING_ITERATIONS 20000
#define NET_PRINTOUT_PERIOD 1000

struct uneural_network network;
fix16_t *net_storage;

static void create_network(void)
{
	int result;

	result = uneural_network_add_input_layer(&network, &input_layer);
	if (result) {
		printf("Error adding input layer\n");
		exit(-1);
	}

	result = uneural_network_add_hidden_layer(&network, &hidden_layer);
	if (result) {
		printf("Error adding hidden layer\n");
		exit(-1);
	}

	result = uneural_network_add_output_layer(&network, &output_layer);
	if (result) {
		printf("Error adding output layer\n");
		exit(-1);
	}

	/* Calculate the amount of data storage we'll need for the
	 * network, initialize and attach it */

	ssize_t storage_required = uneural_network_get_data_requirement(&network);

	if (storage_required < 0) {
		printf("Unable to allocate storage\n");
		exit(-1);
	}

	net_storage = malloc(storage_required);

	if (net_storage == NULL) {
		printf("Unable to allocate network storage!!\n");
		exit(-1);
	}

	printf("Allocated %zd bytes of storage for the network\n", storage_required);

	/* Initialize */
	uneural_network_init_storage(net_storage, storage_required);

	printf("Successfully initialized storage\n");

	/* Link the training data storage to the network */
	uneural_network_data_attach(&network, net_storage, storage_required);

	printf("Successfully attached data\n");

	/* Assign the layer types for hidden and output layers (this only
	 * needs to be done on initial creation). Input layers are
	 * excluded as they are effectively pass-through */

	result = uneural_network_set_layer_type(&hidden_layer, ACTIVATION_TYPE);
	if (result) {
		printf("Error setting hidden layer type\n");
		exit(-1);
	}

	result = uneural_network_set_layer_type(&output_layer, NEURON_TYPE_SIGMOID);
	if (result) {
		printf("Error setting output layer type\n");
		exit(-1);
	}

	printf("Randomizing Network Weights\n");
	/* Randomize the starting weights */
	result = uneural_network_randomize_weights(&network);

	printf("Successfully created raw uNeural Network\n");
}

int main(int argc, char **argv)
{

	create_network();

	fix16_t inputs[4][3] = {
		{F16(0), F16(0), F16(1)},
		{F16(0), F16(1), F16(1)},
		{F16(1), F16(0), F16(1)},
		{F16(1), F16(1), F16(1)},
	};

	fix16_t expected_outputs[4][1] = {
		{F16(0)},
		{F16(1)},
		{F16(1)},
		{F16(0)},
	};

	fix16_t actual_output;

	/* Create training scratchpad space */
	ssize_t scratch_size = uneural_network_get_training_scratch_size(&network);

	if (scratch_size < 0) {
		printf("Unable to get scratch size\n");
		exit(-1);
	}


	fix16_t *training_scratch = malloc(scratch_size);

	if (training_scratch == NULL) {
		printf("Unable to allocate scratch area\n");
		exit(-1);
	}
	printf("Allocated %zd bytes of scratch space\n", scratch_size);

	for (int i = 0; i < 4; i++) {
		uneural_activate_network(&network,
					 inputs[i],
					 &actual_output);
		printf("Initial expected/output %d - %f|%f \n", i,
		       fix16_to_float(expected_outputs[i][0]),
		       fix16_to_float(actual_output));
	}

	/* Train the neural network */

	for (int i = 0; i < TRAINING_ITERATIONS; i++) {
		fix16_t global_error = 0;
		for (int j = 0; j < 4; j++) {
			fix16_t local_error = 0;
			uneural_network_backprop(&network,
						 inputs[j],
						 expected_outputs[j],
						 LEARNING_RATE,
						 training_scratch,
						 &local_error);

			global_error = fix16_add(global_error,
						 fix16_sq(local_error));
		}

		if ((i % NET_PRINTOUT_PERIOD) == 0) {
			printf("Iteration: %d\n", i);
			printf("Global Error: %f\n", fix16_to_float(global_error));
			fix16_t rmse = fix16_sqrt(fix16_sdiv(global_error, F16(4)));
			printf("RMSE: %f\n", fix16_to_float(rmse));

			for (int j = 0; j < 4; j++) {
				uneural_activate_network(&network,
							 inputs[j],
							 &actual_output);
				printf("Actual output %d - %f \n", j, fix16_to_float(actual_output));
			}
		}

	}


}
