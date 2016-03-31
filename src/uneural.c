#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include <stdlib.h>

#include <stdio.h>

#include <uneural.h>


fix16_t uneural_activate_sigmoid(fix16_t sum)
{
    /* Applies the sigmoid activation function to sum and returns the result */
    /* 1 / 1 (e ^-x) */
    fix16_t temp;
    temp = fix16_smul(sum,
                     F16(-1));

    temp = fix16_exp(temp);
    temp = fix16_sadd(fix16_one, temp);
    temp = fix16_sdiv(fix16_one, temp);

    return temp;

}

fix16_t uneural_activate_tanh(fix16_t sum)
{
    /* Applies the tanh activation function to sum and returns the result */
    /* 2*sigmoid(2*x) - 1 */
    fix16_t temp;

    temp = fix16_smul(sum, F16(2));
    temp = uneural_activate_sigmoid(temp);
    temp = fix16_smul(temp, F16(2));
    temp = fix16_ssub(temp, F16(1));
    return temp;
}

fix16_t uneural_activate_relu(fix16_t sum)
{
    /* Applies the ReLU activation function to sum and returns the result */
    /* max(0, x) */
    return fix16_max(F16(0), sum);
}

fix16_t uneural_activate_leaky_relu(fix16_t sum)
{
    /* Applies the Leaky ReLU activation function to sum and returns the result */
    /* TODO: Make leaky arg (a) configurable */
    /* max(x * -a, x) */
    fix16_t lh_arg = fix16_smul(F16(.01), sum);
    return fix16_max(lh_arg, sum);
}

int uneural_activate_layer(struct uneural_layer *work_layer)
{

    if (work_layer == NULL) {
        return -NULL_ARG;
    }

    for (int i = 0; i < work_layer->num_neurons; i++) {

        fix16_t temp = 0;
        struct uneural_neuron *work_neuron = &work_layer->neurons[i];

	/* Clear the work neuron's output */
	work_neuron->output = 0;

        /* Assign the sum of products of the inputs * weights to the
         * neuron's output */
        for (int j = 0; j < work_layer->prev->num_neurons; j++) {
            temp = fix16_smul(work_neuron->weights[j],
			      work_layer->prev->neurons[j].output);

            work_neuron->output = fix16_sadd(work_neuron->output, temp);
        }

        /* Add the neuron's bias */
        work_neuron->output = fix16_sadd(work_neuron->bias[0],
					 work_neuron->output);

        /* Fire the correct activation function for the neuron's type */
        switch (*work_neuron->n_type) {
        case NEURON_TYPE_SIGMOID:
            work_neuron->output = uneural_activate_sigmoid(work_neuron->output);
            break;
        case NEURON_TYPE_TANH:
            work_neuron->output = uneural_activate_tanh(work_neuron->output);
            break;
        case NEURON_TYPE_RELU:
            work_neuron->output = uneural_activate_relu(work_neuron->output);
            break;
        case NEURON_TYPE_LEAKY_RELU:
            work_neuron->output = uneural_activate_leaky_relu(work_neuron->output);
            break;
        default:
            return -1;
        }
    }

    return 0;

}

int uneural_activate_network(struct uneural_network *n,
                             const fix16_t *inputs,
                             fix16_t *outputs)
{
    if (n == NULL || inputs == NULL) {
        return -NULL_ARG;
    }

    if (n->input == NULL) {
        return -MISSING_INPUT_LAYER;
    }

    if (n->output == NULL) {
        return -MISSING_OUTPUT_LAYER;
    }

    /* Move the inputs to the ouput of the input layer (input layer
     * applies no bias or weight on its own, so it's just a
     * passthrough) */
    for (int i = 0; i < n->input->num_neurons; i++) {
        n->input->neurons[i].output = inputs[i];
    }


    /* Activate each layer in turn. Continue until the output layer is
     * reached, then copy the final layer's outputs to the output
     * holding buffer (assuming non-null) */
    struct uneural_layer *work_layer = n->input->next;

    while (work_layer != NULL) {
        int result = uneural_activate_layer(work_layer);

        if (result) {
            return result;
        }
        work_layer = work_layer->next;
    }

    if (outputs != NULL) {
        for (int i = 0; i < n->output->num_neurons; i++) {
            outputs[i] = n->output->neurons[i].output;
        }
    }

    return 0;
}

static struct uneural_layer *uneural_network_last_layer(struct uneural_network *n)
{
    struct uneural_layer *work_layer = n->input;

    while (work_layer != NULL && work_layer->next != NULL) {
        work_layer = work_layer->next;
    }

    return work_layer;
}

int uneural_network_add_input_layer(struct uneural_network *n,
                                    struct uneural_layer *l)
{
    if (n == NULL || l == NULL) {
        return -NULL_ARG;
    }

    if (n->input != NULL) {
        return -INPUT_LAYER_EXISTS;
    }

    n->input = l;

    if (n->output != NULL) {
        l->next = n->output;
    } else {
        l->next = NULL;
    }

    l->prev = NULL;

    return 0;
}

int uneural_network_set_layer_type(struct uneural_layer *l,
                                   enum neuron_type n_type)
{
    if (l == NULL) {
        return -NULL_ARG;
    }

    for (int i = 0; i < l->num_neurons; i++) {
        *l->neurons[i].n_type = (uint32_t)n_type;
    }


    return 0;
}


int uneural_network_add_output_layer(struct uneural_network *n,
                                     struct uneural_layer *l)
{
    if (n == NULL || l == NULL) {
        return -NULL_ARG;
    }

    if (n->output != NULL) {
        return -OUTPUT_LAYER_EXISTS;
    }

    n->output = l;

    /* Link the lists together */
    n->output->prev = uneural_network_last_layer(n);
    n->output->prev->next = l;

    n->output->next = NULL;

    return 0;
}

int uneural_network_add_hidden_layer(struct uneural_network *n,
                                     struct uneural_layer *l)
{
    if (n == NULL || l == NULL) {
        return -NULL_ARG;
    }

    if (n->input == NULL) {
        return -MISSING_INPUT_LAYER;
    }

    /* Walk to the end of the list of hidden layers, and insert the provided
     * layer  */

    struct uneural_layer *last = uneural_network_last_layer(n);

    /* If the last layer is the output layer, put this guy in
     * between, otherwise, just stick this guy at the end and doubly
     * link the list */
    if (last == n->output) {
        l->next = last;
        l->prev = last->prev;
        l->prev->next = l;
        last->prev = l;
    } else {
        last->next = l;
        l->prev = last;
    }

    return 0;
}
