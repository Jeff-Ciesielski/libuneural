#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <uneural.h>

static fix16_t uneural_sigmoid_deriv(fix16_t v)
{
    fix16_t deriv = 0;
    deriv = fix16_sub(F16(1), v);
    deriv = fix16_mul(v, deriv);

    return deriv;

}

static fix16_t uneural_tanh_deriv(fix16_t v)
{
    fix16_t deriv = 0;
    deriv = fix16_sq(v);
    deriv = fix16_sub(F16(1), deriv);
    return deriv;
}

static fix16_t uneural_relu_deriv(fix16_t v)
{
    fix16_t deriv = 0;
    return deriv;

}

static fix16_t uneural_leaky_relu_deriv(fix16_t v)
{
    fix16_t deriv = 0;
    return deriv;

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
        return res;
    }

    uint32_t max_layer_size = uneural_network_largest_layer_size(n);
    uint32_t step_size = (max_layer_size * max_layer_size) * sizeof(fix16_t);

    fix16_t *l1_output = scratch + step_size;
    fix16_t *l1_error = l1_output + step_size;
    fix16_t *l1_delta = l1_error + step_size;
    fix16_t *l1_start_weight = l1_delta + step_size;
    fix16_t *l2_output = l1_start_weight + step_size;
    fix16_t *l2_error = l2_output + step_size;
    fix16_t *l2_delta = l2_error + step_size;
    fix16_t *l2_start_weight = l2_delta + step_size;

    for (struct uneural_layer *l = n->output; l->prev != NULL; l = l->prev) {

        struct uneural_layer *l_p = l->prev;

        if (l == n->output) {
            //printf("Output Layer\n");
            for (int i = 0; i < l->num_neurons; i++) {

                /* Copy in the initial weights */
                for (int j = 0; j < l_p->num_neurons; j++) {
                    l2_start_weight[l->num_neurons * i + j] = l->neurons[i].weights[j];
                }

                l2_output[i] = l->neurons[i].output;
                l2_error[i] = fix16_sub(expected_output[i],
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

                l2_delta[i] = fix16_mul(l2_error[i], deriv);
            }
            
        } else {
            //printf("Hidden Layer\n");
        }

        for (int i = 0; i < l_p->num_neurons; i++) {

            if (l_p->prev != NULL) {
                /* Copy in the initial weights */
                for (int j = 0; j < l_p->prev->num_neurons; j++) {
                    l1_start_weight[l_p->num_neurons * i + j] = l_p->neurons[i].weights[j];
                }
            }

            l1_output[i] = l_p->neurons[i].output;

            for (int j = 0; j < l->num_neurons; j++) {
                fix16_t temp = fix16_mul(l2_delta[j],
                                         l2_start_weight[l->num_neurons * j + i]);

                l1_error[i] = fix16_add(l1_error[i],
                                        temp);
            }

            fix16_t deriv = 0;
            deriv = fix16_sub(F16(1), l1_output[i]);
            deriv = fix16_mul(l1_output[i], deriv);

            l1_delta[i] = fix16_mul(l1_error[i], deriv);
        }

        /* Update working layer weights */
        //printf("Updating layer weights\n");
        for (int i = 0; i < l->num_neurons; i++) {
            for (int j = 0; j < l_p->num_neurons; j++) {
                fix16_t layer_adj = fix16_mul(l2_delta[i],
                                              l1_output[j]);
                layer_adj = fix16_mul(layer_adj,
                                      training_rate);
                //printf("Layer[%d, %d] adj: %f\n", i, j, fix16_to_float(layer_adj));
                l->neurons[i].weights[j] = fix16_add(l->neurons[i].weights[j],
                                                     layer_adj);
            }
        }

        memcpy(l2_output, l1_output, step_size);
        memcpy(l2_error, l1_error, step_size);
        memcpy(l2_delta, l1_delta, step_size);
        memcpy(l2_start_weight, l1_start_weight, step_size);
        memset(l1_output, 0x00, step_size);
        memset(l1_error, 0x00, step_size);
        memset(l1_delta, 0x00, step_size);
        memset(l1_start_weight, 0x00, step_size);
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
