using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralUI.Framework;

namespace NeuralUI.ViewModels
{
    public class NeuronViewModel : ViewModelBase
    {
        private double top;
        private double left;
        private string _value;

        public double Top
        {
            get => top;
            set => SetProperty(ref top, value);
        }

        public double Left
        {
            get => left;
            set => SetProperty(ref left, value);
        }

        public string Value
        {
            get => _value;
            set => SetProperty(ref _value, value);
        }
    }
}
