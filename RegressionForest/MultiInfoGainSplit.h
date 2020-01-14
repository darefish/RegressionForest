#pragma once
#include "BaseSplitMethod.h"

namespace hiveRegressionForest
{
	class CMultiInfoGainSpliter : public INodeSpliter
	{
	public:
		CMultiInfoGainSpliter();
		~CMultiInfoGainSpliter();

	protected:
		virtual void _generateSortedFeatureResponsePairSetV(const std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet) override;

	private:
		virtual void __findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap) override;
	};
}