#pragma once
#include "Node.h"
#include "math/RegressionAnalysisInterface.h"
#include <boost/serialization/base_object.hpp>

namespace hiveRegressionForest
{
	class CSingleResponseNode : public CNode
	{
	public:
		CSingleResponseNode();
		CSingleResponseNode(unsigned int vLevel) { m_Level = vLevel; }
		~CSingleResponseNode();

		virtual void createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) override;
		virtual float predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex) const override;

	protected:
		virtual float _getNodeVarianceV(unsigned int vResponseIndex = 0) const override;

	private:
		float m_NodeVariance = 0.0f;
		hiveRegressionAnalysis::IBaseRegression* m_pRegressionModel = nullptr;

	private:
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & boost::serialization::base_object<CNode>(*this);

			ar & m_NodeVariance;
			ar & m_pRegressionModel;
		}

		friend class boost::serialization::access;
	};
}
