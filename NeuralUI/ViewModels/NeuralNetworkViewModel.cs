using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using NeuralUI.Framework;

namespace NeuralUI.ViewModels
{
    public class NeuralNetworkViewModel : ViewModelBase
    {
        private ObservableCollection<NeuronViewModel> neurons;

        public NeuralNetworkViewModel(Perceptron perceptron)
        {
            var x = 10.0;

            var temp = new ObservableCollection<NeuronViewModel>();
            foreach (var layer in perceptron.Layers)
            {
                var y = 10.0;
                foreach (var neuron in layer.Neurons)
                {
                    var vm = new NeuronViewModel
                    {
                        Left = x,
                        Top = y,
                        Value = neuron.Bias.ToString(CultureInfo.InvariantCulture),
                    };
                    temp.Add(vm);
                    y += 80;
                }

                x += 80;
            }

            Neurons = temp;
        }

        public ObservableCollection<NeuronViewModel> Neurons
        {
            get => neurons;
            set => SetProperty(ref neurons, value);
        }
    }
}
